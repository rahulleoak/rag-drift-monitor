"""Changed facts in updated documents: the check for the blind spot in D27.

Embedding drift barely moves when an updated document changes only a
number ("killed over 75 million" -> "over 20 million"; F17). This check
compares an updated document with its previous version sentence by
sentence: each new or rewritten sentence is matched with the most similar
sentence of the previous version in the same section, and if their numbers
differ it is reported as a changed fact, with both sentences as evidence.

Formatting-only differences ("2217" vs "2,217") are ignored. Pure
functions, no LLM. Applies only to documents that are edited over time.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d(?:[\d,.]*\d)?")
_WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class FactChange:
    """One sentence whose numbers changed between two versions."""

    section: str
    before: str
    after: str
    removed: tuple[str, ...]   # numbers only in the old sentence
    added: tuple[str, ...]     # numbers only in the new sentence
    similarity: float          # word overlap of the matched sentences


def _sentences(text: str) -> list[str]:
    return [" ".join(s.split()) for para in text.split("\n") for s in _SENT.split(para) if s.strip()]


def numbers(sentence: str) -> list[str]:
    """Numbers in a sentence, normalised so formatting does not count
    ("2,217" -> "2217"; a trailing full stop is not part of the number)."""
    return [n.replace(",", "").rstrip(".") for n in _NUMBER.findall(sentence)]


def _words(sentence: str) -> set[str]:
    return set(_WORD.findall(sentence.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def fact_changes(old_sections: Sequence[tuple[str, str]], new_sections: Sequence[tuple[str, str]],
                 min_similarity: float = 0.6) -> list[FactChange]:
    """Changed facts between two versions of a document, given as
    (heading, text) sections.

    A new sentence that is not in the old version is matched with the old
    sentence, in the same section (or anywhere, if the section is new),
    that shares the most words (numbers aside). If the overlap is at least
    ``min_similarity`` and the two sentences' numbers differ, it is a
    changed fact. Lower overlap means a rewrite or a new statement, not a
    changed fact.
    """
    old_by_section = {h: _sentences(t) for h, t in old_sections}
    old_all = [s for sents in old_by_section.values() for s in sents]
    old_set = set(old_all)
    out = []
    for heading, text in new_sections:
        candidates = old_by_section.get(heading) or old_all
        cand_words = [(s, _words(s)) for s in candidates]
        for sent in _sentences(text):
            if sent in old_set:
                continue
            new_nums = numbers(sent)
            if not new_nums:
                continue
            w = _words(sent)
            best, score = None, 0.0
            for s, sw in cand_words:
                j = _jaccard(w, sw)
                if j > score:
                    best, score = s, j
            if best is None or score < min_similarity:
                continue
            old_nums = numbers(best)
            if sorted(old_nums) == sorted(new_nums):
                continue                               # same numbers: wording or formatting only
            removed = tuple(n for n in dict.fromkeys(old_nums) if n not in new_nums)
            added = tuple(n for n in dict.fromkeys(new_nums) if n not in old_nums)
            if not _replaced(removed, added) or _is_reference(sent) or _is_reference(best):
                continue
            out.append(FactChange(heading, best, sent, removed, added, round(score, 3)))
    return out


_REFERENCE = re.compile(r"\bISBN\b|\bISSN\b|\bdoi\b|\bRetrieved\b|\bpp?\.\s*\d|\bVol\.|\d+px\b|\bthumb(?:nail)?\b|\bupright\b", re.I)   # citations, image captions


def _is_reference(sentence: str) -> bool:
    """Citation lines and fragments: their numbers are not facts."""
    return bool(_REFERENCE.search(sentence)) or len(_WORD.findall(sentence.lower())) < 6


def _replaced(removed: Sequence[str], added: Sequence[str]) -> bool:
    """A value was replaced: some number removed and another added.

    A number only added ("In 2016, a study ...") is new context, and one only
    removed is usually a dropped citation or qualifier; neither changes what
    the sentence states. Pairs where one number contains the other ("1893"
    and "18939") are text-cleaning leftovers, not changes.
    """
    real = [(r, a) for r in removed for a in added if not (r in a or a in r)]
    return bool(removed and added and real)
