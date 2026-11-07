"""The only module that calls an LLM (D9, D15).

Every request goes through ``LLMClient``, which:
    - asks for **structured output** (a JSON schema from a pydantic model) and
      validates the reply against it;
    - **caches** every response on disk, keyed on model, prompt version,
      system prompt, user prompt, schema and settings, so a rerun never pays
      twice and results are reproducible (current models reject a
      temperature setting, so the cache is what makes runs repeatable);
    - **enforces the budget**: before any spend it estimates the cost and
      refuses if the running total in the cost ledger would exceed
      ``llm.budget_usd``; after each response it records the actual cost;
    - **handles refusals**: live calls use the API's server-side fallback
      (``fallbacks="default"``); the Batch API does not support it, so a
      batch refusal is retried live with fallback; anything still refused is
      returned as refused, never silently dropped;
    - **does not cache broken replies** (cut off, or not matching the schema),
      so a rerun retries them; their cost is still recorded;
    - runs either **live** (one request at a time) or as a **batch** (half
      price, results within hours);
    - makes batches **resumable**: the batch id and its request list are saved
      the moment a batch is created (``<cache>/_batches/``), so if the run is
      interrupted while waiting, rerunning the same command collects that
      batch instead of paying for a new one.

The LLM writes questions, answers, judgements and explanations; it never
makes the decisions (those live in tested Python code).
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

FALLBACK_BETA = "server-side-fallback-2026-07-01"
LEDGER_FIELDS = ["time", "job", "model", "mode", "input_tokens", "output_tokens", "cost_usd",
                 "stop_reason", "key"]


def api_key(env_file: Path | None = None) -> str | None:
    """The Anthropic key: the environment's if it is non-empty, else the project's ``.env``.

    An environment variable that exists but is empty would otherwise hide the
    ``.env`` value (python-dotenv never overrides existing variables), and the
    environment is left unchanged either way. None lets the SDK fall back to
    its other credential sources.
    """
    import os

    from dotenv import dotenv_values

    from driftmon.config import REPO_ROOT
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    path = env_file or REPO_ROOT / ".env"
    return (dotenv_values(path).get("ANTHROPIC_API_KEY") or "").strip() or None if path.exists() else None


class BudgetExceeded(RuntimeError):
    """Raised before a request whose estimated cost would pass the budget."""


@dataclass(frozen=True)
class LLMRequest:
    """One structured-output request.

    Args:
        job: which job this belongs to (``llm.jobs`` in the config), e.g. "questions".
        model: model id.
        system: system prompt (stable across a job, so it caches well).
        prompt: the user message.
        schema: pydantic model the reply must match.
        prompt_version: bump when a prompt changes, so cached answers are not reused.
        effort: thinking effort ("low", "medium", "high").
        max_tokens: output cap; thinking counts towards it.
        replicate: 0 for the first answer; 1, 2, ... for independent repeats of
            the same request (used to review twice, D19). Repeats get their own
            cache entries; replicate 0 keeps the original cache key.
    """

    job: str
    model: str
    system: str
    prompt: str
    schema: type[BaseModel]
    prompt_version: str
    effort: str = "medium"
    max_tokens: int = 4000
    replicate: int = 0

    @property
    def key(self) -> str:
        fields = {"model": self.model, "system": self.system, "prompt": self.prompt,
                  "schema": self.schema.model_json_schema(), "version": self.prompt_version,
                  "effort": self.effort, "max_tokens": self.max_tokens}
        if self.replicate:  # only repeats change the key, so existing caches stay valid
            fields["replicate"] = self.replicate
        payload = json.dumps(fields, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LLMResult:
    """The outcome of one request."""

    key: str
    parsed: BaseModel | None          # None if refused or the reply did not validate
    text: str
    stop_reason: str
    model: str                        # the model that actually answered (may be a fallback)
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The pydantic model as a strict JSON schema: every property required, no extras."""
    schema = model.model_json_schema()

    def tighten(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in node.values():
                tighten(v)
        elif isinstance(node, list):
            for v in node:
                tighten(v)

    tighten(schema)
    return schema


@dataclass
class LLMClient:
    """Cached, budgeted, structured LLM calls.

    Args:
        config: the project config (``llm`` section).
        cache_dir: where responses are cached (``<cache>/llm``).
        ledger: CSV file recording the cost of every billed request.
        client: an ``anthropic.Anthropic`` client, or a stand-in with the same
            methods (tests); created from ``ANTHROPIC_API_KEY`` if omitted.
        sleep: wait function used while polling batches (replaced in tests).
        log: progress messages while a batch runs (e.g. ``print``); silent by default.
    """

    config: Mapping[str, Any]
    cache_dir: Path
    ledger: Path
    client: Any = None
    sleep: Callable[[float], None] = time.sleep
    poll_seconds: float = 30.0
    log: Callable[[str], None] = lambda message: None
    _spent: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.ledger = Path(self.ledger)
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key(),
                                              max_retries=int(self.config.get("max_retries", 4)))

    # ---------- money ----------

    @property
    def budget(self) -> float:
        return float(self.config.get("budget_usd") or 0.0)

    def spent(self) -> float:
        """Total recorded spend so far (all jobs, all runs)."""
        if self._spent is None:
            self._spent = 0.0
            if self.ledger.exists():
                with self.ledger.open(encoding="utf-8") as f:
                    self._spent = sum(float(r["cost_usd"]) for r in csv.DictReader(f))
        return self._spent

    def cost(self, model: str, input_tokens: int, output_tokens: int, batch: bool) -> float:
        price = self.config["prices"][model]
        usd = (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6
        return usd * (0.5 if batch else 1.0)

    def estimate(self, requests: Sequence[LLMRequest], batch: bool) -> float:
        """Conservative cost estimate: ~4 characters per input token, plus the
        job's expected output (including thinking)."""
        total = 0.0
        for r in requests:
            expected = int(self.config.get("jobs", {}).get(r.job, {}).get("expected_output", r.max_tokens))
            total += self.cost(r.model, (len(r.system) + len(r.prompt)) // 4 + 50, expected, batch)
        return total

    def _check_budget(self, requests: Sequence[LLMRequest], batch: bool) -> None:
        if not requests:
            return
        est = self.estimate(requests, batch)
        if self.spent() + est > self.budget:
            raise BudgetExceeded(f"estimated ${est:.2f} for {len(requests)} request(s) would take spend "
                                 f"from ${self.spent():.2f} past the ${self.budget:.2f} budget")

    def _record(self, job: str, model: str, mode: str, res: LLMResult) -> None:
        new = not self.ledger.exists()
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "job": job,
                        "model": model, "mode": mode, "input_tokens": res.input_tokens,
                        "output_tokens": res.output_tokens, "cost_usd": f"{res.cost_usd:.6f}",
                        "stop_reason": res.stop_reason, "key": res.key})
        self._spent = self.spent() + res.cost_usd

    # ---------- cache ----------

    def _path(self, req: LLMRequest) -> Path:
        return self.cache_dir / req.job / f"{req.key}.json"

    def cached(self, req: LLMRequest) -> LLMResult | None:
        """The cached result for this exact request, if any (marked ``cached``)."""
        p = self._path(req)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        parsed = req.schema.model_validate(d["parsed"]) if d.get("parsed") is not None else None
        return LLMResult(key=req.key, parsed=parsed, text=d["text"], stop_reason=d["stop_reason"],
                         model=d["model"], input_tokens=d["input_tokens"], output_tokens=d["output_tokens"],
                         cost_usd=0.0, cached=True, error=d.get("error", ""))

    def _store(self, req: LLMRequest, res: LLMResult) -> None:
        p = self._path(req)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "job": req.job, "requested_model": req.model, "prompt_version": req.prompt_version,
            "model": res.model, "stop_reason": res.stop_reason, "text": res.text,
            "parsed": res.parsed.model_dump() if res.parsed is not None else None,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "cost_usd": res.cost_usd, "error": res.error,
            "prompt": req.prompt, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=1), encoding="utf-8")

    # ---------- requests ----------

    def _params(self, req: LLMRequest) -> dict[str, Any]:
        return {"model": req.model, "max_tokens": req.max_tokens,
                "system": [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": req.prompt}],
                "output_config": {"effort": req.effort,
                                  "format": {"type": "json_schema", "schema": json_schema(req.schema)}}}

    def _result(self, req: LLMRequest, msg: Any, batch: bool) -> LLMResult:
        """Turn an API message into an LLMResult (validated, costed)."""
        text = next((b.text for b in msg.content if getattr(b, "type", "") == "text"), "")
        usage = msg.usage
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0) + \
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0) + \
            int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        model = getattr(msg, "model", req.model) or req.model
        price_model = model if model in self.config["prices"] else req.model
        res = LLMResult(key=req.key, parsed=None, text=text, stop_reason=msg.stop_reason or "",
                        model=model, input_tokens=tokens_in, output_tokens=tokens_out,
                        cost_usd=self.cost(price_model, tokens_in, tokens_out, batch))
        if res.stop_reason == "refusal":
            res.error = "refused"
        elif res.stop_reason == "max_tokens":
            res.error = "output cut off at max_tokens"
        else:
            try:
                res.parsed = req.schema.model_validate_json(text)
            except ValidationError as e:
                res.error = f"invalid output: {e.errors()[0]['msg'] if e.errors() else e}"
        return res

    def complete(self, req: LLMRequest) -> LLMResult:
        """One live request (with server-side fallback on refusal), cached."""
        hit = self.cached(req)
        if hit is not None:
            return hit
        self._check_budget([req], batch=False)
        msg = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default",
                                               **self._params(req))
        res = self._result(req, msg, batch=False)
        self._record(req.job, res.model, "live", res)
        if res.ok or res.error == "refused":  # a live refusal already had its fallback: keep it
            self._store(req, res)             # broken replies are not cached, so a rerun retries
        return res

    def complete_many(self, requests: Sequence[LLMRequest], batch: bool | None = None,
                      on_submit: Callable[[str], None] | None = None) -> list[LLMResult]:
        """Many requests: cached ones are free; the rest run live or as one batch.

        Results come back in the order of ``requests``. ``on_submit`` receives
        the batch id, so a long-running batch can be resumed if interrupted.
        """
        batch = bool(self.config.get("use_batch_api", True)) if batch is None else batch
        results: dict[str, LLMResult] = {}
        todo: dict[str, LLMRequest] = {}
        for r in requests:
            hit = self.cached(r)
            if hit is not None:
                results[r.key] = hit
            else:
                todo.setdefault(r.key, r)
        if todo:
            self._check_budget(list(todo.values()), batch=batch)
            if batch:
                results.update(self._run_batch(list(todo.values()), on_submit))
                # the Batch API has no fallback: retry its refusals live, with fallback
                for key, res in list(results.items()):
                    if key in todo and res.error == "refused":
                        results[key] = self.complete(todo[key])
            else:
                for r in todo.values():
                    results[r.key] = self.complete(r)
        return [results[r.key] for r in requests]

    def _run_batch(self, requests: list[LLMRequest], on_submit) -> dict[str, LLMResult]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        by_key = {r.key: r for r in requests}
        # the same set of uncached requests always maps to the same pending file
        fingerprint = hashlib.sha256("\n".join(sorted(by_key)).encode()).hexdigest()[:20]
        pending = self.cache_dir / "_batches" / f"{fingerprint}.json"
        if pending.exists():  # an earlier run submitted this batch and was interrupted
            saved = json.loads(pending.read_text(encoding="utf-8"))
            batch_id = saved["batch_id"]
            by_id = {cid: by_key[key] for cid, key in saved["custom_ids"].items()}
            self.log(f"Resuming batch {batch_id} ({len(by_id)} requests, submitted {saved['created']})")
        else:
            by_id = {f"r{i}": r for i, r in enumerate(requests)}
            created = self.client.messages.batches.create(requests=[
                Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**self._params(r)))
                for cid, r in by_id.items()])
            batch_id = created.id
            pending.parent.mkdir(parents=True, exist_ok=True)
            pending.write_text(json.dumps({
                "batch_id": batch_id, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "jobs": sorted({r.job for r in requests}),
                "custom_ids": {cid: r.key for cid, r in by_id.items()}}, indent=1), encoding="utf-8")
            self.log(f"Submitted batch {batch_id} ({len(by_id)} requests); saved to {pending}")
        if on_submit:
            on_submit(batch_id)
        out = self.collect_batch(batch_id, by_id)
        pending.unlink(missing_ok=True)  # collected and cached: nothing left to resume
        return out

    def collect_batch(self, batch_id: str, by_id: Mapping[str, LLMRequest]) -> dict[str, LLMResult]:
        """Wait for a batch, then cache and record every result (by custom id)."""
        waited = 0.0
        while (status := self.client.messages.batches.retrieve(batch_id)).processing_status != "ended":
            counts = getattr(status, "request_counts", None)
            done = (f"{counts.succeeded + counts.errored + counts.canceled + counts.expired} of "
                    f"{len(by_id)} done" if counts else "in progress")
            if waited % 300 == 0:  # every 5 minutes
                self.log(f"Batch {batch_id}: {done} after {waited / 60:.0f} min")
            self.sleep(self.poll_seconds)
            waited += self.poll_seconds
        out: dict[str, LLMResult] = {}
        for item in self.client.messages.batches.results(batch_id):
            req = by_id[item.custom_id]
            kind = item.result.type
            if kind == "succeeded":
                res = self._result(req, item.result.message, batch=True)
                self._record(req.job, res.model, "batch", res)
                if not res.ok:  # refused or broken: not cached (refusals are retried live)
                    out[req.key] = res
                    continue
            else:  # errored / canceled / expired: not billed; not cached, so a rerun retries it
                res = LLMResult(key=req.key, parsed=None, text="", stop_reason=kind, model=req.model,
                                input_tokens=0, output_tokens=0, cost_usd=0.0, error=f"batch {kind}")
                out[req.key] = res
                continue
            self._store(req, res)
            out[req.key] = res
        return out
