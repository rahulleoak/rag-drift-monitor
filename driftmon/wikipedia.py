"""Long-form documents: Wikipedia articles as they stood at the end of each period (D26).

Articles are a stratified random sample of Wikipedia's Level 3 "vital
articles" list (about 1,000 broad topics in 11 sections), plus a small,
separately labelled set of articles tied to known events, as the
known-drift check. Each article is captured at its last revision before the
end of every period, so the same document can be compared with itself over
time. Pages are followed by page id, so renames (e.g. "Global warming" to
"Climate change") do not break the history.

Network access is limited to the public MediaWiki API; every response is
cached on disk, so a rerun makes no requests. Parsing and cleaning are pure
functions.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import mwparserfromhell as mw
import numpy as np

_HEADING = re.compile(r"^==([^=].*?)==\s*$", re.M)
_LINK = re.compile(r"\[\[([^\]|#]+)")
_DROP_LINK_PREFIXES = ("file:", "image:", "category:", "media:")


# ---------- the vital-articles list ----------

def vital_titles(wikitext: str) -> dict[str, list[str]]:
    """Titles listed under each top-level section of a vital-articles page.

    Only bulleted list items count (the page's prose also links to articles).
    """
    out: dict[str, list[str]] = {}
    parts = _HEADING.split(wikitext)
    for heading, body in zip(parts[1::2], parts[2::2]):
        titles = [m.group(1).strip() for line in body.splitlines() if line.lstrip().startswith("*")
                  for m in [_LINK.search(line)] if m]
        if titles:
            out[heading.strip()] = list(dict.fromkeys(titles))
    return out


def sample_titles(by_section: dict[str, list[str]], per_section: int, seed: int) -> list[tuple[str, str]]:
    """``per_section`` random titles from every section: (section, title)."""
    rng = np.random.default_rng(seed)
    picks = []
    for section in sorted(by_section):
        titles = sorted(by_section[section])
        for i in sorted(rng.permutation(len(titles))[:per_section]):
            picks.append((section, titles[i]))
    return picks


# ---------- wikitext to clean sections ----------

def _clean(code: mw.wikicode.Wikicode) -> str:
    """Plain text from wikitext: no templates (infoboxes, citations), tables,
    references, files or categories; link text kept."""
    for node in list(code.filter_wikilinks(recursive=False)):
        if str(node.title).strip().lower().startswith(_DROP_LINK_PREFIXES):
            _remove(code, node)
    for node in list(code.filter_tags(recursive=False)):
        if str(node.tag).lower() in {"table", "ref", "gallery", "math", "references"}:
            _remove(code, node)
    text = code.strip_code(normalize=True, collapse=True, keep_template_params=False)
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        # drop leftovers: bare list markers, table fragments, very short lines that are not prose
        if line.startswith(("{|", "|", "!")) or (line and len(line.split()) < 3 and not line.endswith(".")):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\(\s*(?:[;,]\s*)+", "(", text)     # "( ; ; 428 BC" left by removed templates
    text = re.sub(r"\s*\(\s*\)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _remove(code: mw.wikicode.Wikicode, node: Any) -> None:
    try:
        code.remove(node)
    except ValueError:      # already removed with a parent
        pass


def sections(wikitext: str, skip: Iterable[str]) -> list[tuple[str, str]]:
    """(heading, plain text) for the lead and each section, in order.

    Subsections are their own sections, labelled "Parent / Child". Reference
    and navigation sections (``skip``, matched case-insensitively on the
    top-level heading) are dropped with their subsections.
    """
    skip = {s.lower() for s in skip}
    code = mw.parse(wikitext)
    out, parents = [], []
    for sec in code.get_sections(flat=True, include_lead=True):
        headings = sec.filter_headings(recursive=False)
        if headings:
            h = headings[0]
            title = " ".join(h.title.strip_code().split())
            parents = parents[: max(h.level - 2, 0)] + [title]
            sec.remove(h)
        else:
            parents = ["Lead"]
        if parents[0].lower() in skip:
            continue
        text = _clean(sec)
        if text:
            out.append((" / ".join(parents), text))
    return out


def is_redirect(wikitext: str) -> bool:
    return wikitext.lstrip().lower().startswith("#redirect")


# ---------- the MediaWiki API, cached ----------

class WikiClient:
    """A polite, cached client for the MediaWiki action API.

    One request at a time with a pause between them, ``maxlag`` set (the
    server asks clients to back off when busy), a descriptive User-Agent
    without personal contact details, and every response saved under
    ``cache_dir`` so it is fetched once.
    """

    def __init__(self, api: str, user_agent: str, cache_dir: str | Path, pause_s: float = 0.2):
        import requests

        self.api, self.pause_s = api, pause_s
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.requests = 0

    def get(self, params: dict[str, Any], cache_name: str) -> dict[str, Any]:
        path = self.cache_dir / f"{cache_name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        query = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(6):
            time.sleep(self.pause_s)
            r = self.session.get(self.api, params=query, timeout=60)
            self.requests += 1
            if r.status_code == 200:
                data = r.json()
                if data.get("error", {}).get("code") != "maxlag":
                    break
            time.sleep(min(60, 5 * 2 ** attempt))
        else:
            raise RuntimeError(f"Wikipedia API kept failing for {cache_name}: HTTP {r.status_code}")
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error for {cache_name}: {data['error']}")
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def page_text(self, title: str) -> str:
        """Current wikitext of a page (redirects followed)."""
        data = self.get({"action": "query", "redirects": 1, "prop": "revisions", "rvprop": "content",
                         "rvslots": "main", "titles": title}, _safe(f"page_{title}"))
        return data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]

    def page_ids(self, titles: Sequence[str]) -> dict[str, tuple[int, str] | None]:
        """Each title's (page id, current title) after redirects; None if missing."""
        out: dict[str, tuple[int, str] | None] = {}
        for start in range(0, len(titles), 50):
            batch = list(titles[start:start + 50])
            data = self.get({"action": "query", "redirects": 1, "titles": "|".join(batch)},
                            f"ids_{hashlib.sha256('|'.join(batch).encode()).hexdigest()[:16]}")
            q = data["query"]
            moved = {}
            for key in ("normalized", "redirects"):
                for m in q.get(key, []):
                    moved[m["from"]] = m["to"]
            pages = {p["title"]: p for p in q["pages"]}
            for t in batch:
                final = t
                while final in moved:
                    final = moved[final]
                p = pages.get(final)
                out[t] = None if p is None or p.get("missing") else (int(p["pageid"]), p["title"])
        return out

    def revision_at(self, page_id: int, timestamp: str) -> dict[str, Any] | None:
        """The page's last revision at or before ``timestamp`` (ISO, UTC):
        revid, timestamp, size in bytes and wikitext. None if the page did
        not exist yet."""
        data = self.get({"action": "query", "prop": "revisions", "pageids": page_id, "rvlimit": 1,
                         "rvdir": "older", "rvstart": timestamp, "rvslots": "main",
                         "rvprop": "ids|timestamp|content|size"}, f"rev_{page_id}_{timestamp[:10]}")
        page = data["query"]["pages"][0]
        revs = page.get("revisions") or []
        if not revs:
            return None
        r = revs[0]
        return {"revid": r["revid"], "timestamp": r["timestamp"], "bytes": r.get("size"),
                "content": r["slots"]["main"].get("content", "")}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:150]
