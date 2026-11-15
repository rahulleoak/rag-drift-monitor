"""The truth table: does the four-kinds split catch a worse answer step? (D39)

    python scripts/truth_table.py --estimate   # no LLM spend: the cost of the new cells
    python scripts/truth_table.py              # run it (Batch API: one answer batch, one judge batch)
    python scripts/truth_table.py --live       # same, live calls (faster, twice the price)

Takes a saved pipeline run (default: the newest run that is not a simulated
fault) and answers its answerable questions again from the same top 5,
changing the answer model, the answer prompt, or both (cells in
driftmon/evaluation/truth_table.py). The same judge grades every answer.
The baseline cell is the run's own answers, served from the cache: if any of
it needs a new call, the rebuild does not match the run and the script stops.
Writes to runs/truth_table/:
    answers.csv    every answer in every cell: grades, bucket, length
    summary.csv    per cell and index: the four-kinds counts and the producing share
    verdict.json   per changed cell: isolated? producing caught? (thresholds fixed before results)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftmon.agents.impact_assessment import _placeholder, _run  # noqa: E402
from driftmon.config import load_config, resolve_path  # noqa: E402
from driftmon.dashboard.runs import default_run, list_runs  # noqa: E402
from driftmon.data.corpus import load_corpus  # noqa: E402
from driftmon.data.queries import load_questions  # noqa: E402
from driftmon.evaluation import answers as an  # noqa: E402
from driftmon.evaluation import truth_table as tt  # noqa: E402
from driftmon.llm.client import LLMClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="pipeline run id (default: the newest normal run)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="no LLM spend: print the cost of the new cells")
    mode.add_argument("--live", action="store_true", help="live calls instead of the Batch API")
    args = parser.parse_args(argv)
    cfg = load_config()
    runs_dir = resolve_path(cfg["paths"]["runs"])
    run_id = args.run or default_run(runs_dir, list_runs(runs_dir))
    if not run_id:
        raise SystemExit("No pipeline run found; run scripts/run_pipeline.py first.")
    folder = runs_dir / run_id
    runs = pd.read_csv(folder / "retrieval_runs.csv")
    plan = tt.load_plan(pd.read_csv(folder / "impact_answers.csv"), runs, int(cfg["impact"]["answer_k"]))
    questions = load_questions(cfg)
    docs, _ = load_corpus(cfg)
    print(f"Run {run_id}: {plan['question_id'].nunique()} answerable questions, {len(plan)} answers per cell.")

    batch = not args.live
    llm = LLMClient(config=cfg["llm"], cache_dir=resolve_path(cfg["paths"]["cache"]) / "llm",
                    ledger=runs_dir / "llm_costs.csv", log=lambda m: print(m, flush=True))
    reqs = {c.name: tt.answer_requests(plan, questions, docs, cfg, c) for c in tt.CELLS}
    missing = sum(llm.cached(r) is None for r in reqs[tt.BASELINE])
    if missing:
        raise SystemExit(f"{missing} baseline answers are not in the cache: the rebuilt requests do not match "
                         f"run {run_id}. Stopping before any spend.")

    if args.estimate:
        new = [r for c in tt.CELLS[1:] for r in reqs[c.name] if llm.cached(r) is None]
        a_cost = llm.estimate(new, batch)
        probe = [_placeholder(r) for r in llm.complete_many(reqs[tt.BASELINE], batch=batch)]
        j_cost = llm.estimate([j for j in an.judge_requests(plan, probe, questions, docs, cfg) if j], batch) \
            * (len(tt.CELLS) - 1)
        print(f"Estimate: answers ${a_cost:.2f} ({len(new)} new requests), judging at most ${j_cost:.2f}; total "
              f"${a_cost + j_cost:.2f} ({'batch' if batch else 'live'} price). Spent so far ${llm.spent():.2f}.")
        return 0

    # 1. Answers: every cell's requests in one call (the baseline is cached, so only the new cells run).
    flat = [r for c in tt.CELLS for r in reqs[c.name]]
    res = iter(llm.complete_many(flat, batch=batch))
    answers = {c.name: [next(res) for _ in reqs[c.name]] for c in tt.CELLS}
    # 2. The same judge, every answer, in one call.
    jreqs = {c: an.judge_requests(plan, answers[c], questions, docs, cfg) for c in answers}
    jflat = [r for c in answers for r in jreqs[c]]
    jres = iter(_run(llm, jflat, batch))
    judgements = {c: [next(jres) for _ in jreqs[c]] for c in answers}

    scored = pd.concat([tt.score(plan, answers[c], judgements[c], questions, runs, c) for c in answers],
                       ignore_index=True)
    summary = tt.summarise(scored)
    result = {"run": run_id, "cells": {c.name: c.label for c in tt.CELLS},
              "small_model": cfg["truth_table"]["small_model"], **tt.verdict(scored, cfg)}
    out = runs_dir / "truth_table"
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "answers.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))
    print(json.dumps(result, indent=2))
    failed = sum(not r.ok for v in answers.values() for r in v) + \
        sum(1 for v in judgements.values() for j in v if j is not None and not j.ok)
    print(f"{failed} failed LLM calls (left out, never scored wrong). Spent so far ${llm.spent():.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
