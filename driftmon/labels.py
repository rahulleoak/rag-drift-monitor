"""Hand labels that check the LLM judges (P8).

The author labels a small sample without seeing the LLM's verdict; code then
measures how often they agree. Two parts now, a third with Agent D:
    - questions: was the question fair and its reference answer right? Kept
      questions mostly (to catch systematic reviewer errors, F9), plus some
      the reviewer rejected, so both kinds of mistake can show.
    - relevance: does this article answer the question (0 / 1 / 2)? Drawn
      evenly across the judge's grades, so agreement is measured on each.
    - answers (Agent D): is the answer correct and supported by the
      retrieved articles (0 / 1 / 2)? A separate sheet (``make-answers``).

The sheet the author fills in never holds the LLM verdicts; those are in a
separate key file. Pure functions; no LLM calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from driftmon.embeddings import document_text

SHEET_COLUMNS = ["item_id", "part", "instructions", "question", "reference_answer", "text", "your_label",
                 "your_note"]
INSTRUCTIONS = {
    "questions": "1 = a fair question a journalist could ask, answered by the article, and the reference "
                 "answer is right; 0 = otherwise",
    "relevance": "2 = the article answers the question; 1 = same topic or event but no answer; 0 = not relevant",
    "answers": "2 = gives the fact asked for correctly, and everything it says is in the ARTICLES; "
               "1 = partly right, or adds something the ARTICLES do not say; 0 = wrong, or no answer",
    "facts": "1 = the AFTER sentence states a different fact from BEFORE (a new count, date, amount or ranking); "
             "0 = only wording, rounding, units, formatting or an example changed",
}


def sample_questions(table: pd.DataFrame, kept: int, rejected: int, seed: int) -> pd.DataFrame:
    """Questions to label: ``kept`` accepted ones and ``rejected`` ones the
    reviewer (not the code checks) turned down. Verdict: 1 kept, 0 rejected."""
    rng = np.random.default_rng(seed)
    acc = table[table["set"] == "answerable"]
    by_reviewer = table[(table["set"] == "") & table["rejected_by"].fillna("").str.startswith("review")]
    pick = [acc.iloc[rng.permutation(len(acc))[:kept]].assign(verdict=1),
            by_reviewer.iloc[rng.permutation(len(by_reviewer))[:rejected]].assign(verdict=0)]
    out = pd.concat(pick, ignore_index=True)
    return pd.DataFrame({"part": "questions", "question": out["question"],
                         "reference_answer": out["reference_answer"],
                         "text": [document_text(t, x) for t, x in zip(out["title"], out["text"])],
                         "source_id": out["question_id"], "verdict": out["verdict"]})


def sample_relevance(grades: pd.DataFrame, questions: pd.DataFrame, docs: pd.DataFrame, per_grade: int,
                     seed: int) -> pd.DataFrame:
    """(question, article) pairs to grade, ``per_grade`` for each judge grade."""
    rng = np.random.default_rng(seed)
    g = grades.dropna(subset=["grade"]).astype({"grade": int})
    picks = [grp.iloc[rng.permutation(len(grp))[:per_grade]] for _, grp in g.groupby("grade")]
    out = pd.concat(picks, ignore_index=True).merge(
        questions[["question_id", "question", "reference_answer"]], on="question_id")
    d = docs.set_index("doc_id")
    return pd.DataFrame({"part": "relevance", "question": out["question"],
                         "reference_answer": out["reference_answer"],
                         "text": [f"(published {d.at[i, 'date']}) {document_text(d.at[i, 'title'], d.at[i, 'text'])}"
                                  for i in out["doc_id"]],
                         "source_id": out["question_id"] + "|" + out["doc_id"], "verdict": out["grade"]})


def sample_facts(changes: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Changed facts flagged by the D27 check, to measure its precision.
    Every item is one the check flagged, so the verdict is always 1."""
    pick = changes.sample(n=min(n, len(changes)), random_state=seed)
    return pd.DataFrame({"part": "facts", "question": [f"{t} ({p})" for t, p in zip(pick["title"], pick["period"])],
                         "reference_answer": "",
                         "text": [f"BEFORE: {b}\nAFTER: {a}" for b, a in zip(pick["before"], pick["after"])],
                         "source_id": [f"{i}|{p}" for i, p in zip(pick["page_id"], pick["period"])],
                         "verdict": 1})


def answer_verdict(correctness: float, grounded: float) -> int:
    """The judge's grades on the sheet's 0/1/2 scale: 2 only if correct and
    grounded; an ungrounded correct answer counts as 1."""
    if correctness == 2:
        return 2 if grounded == 1 else 1
    return int(correctness)


def sample_answers(answers: pd.DataFrame, questions: pd.DataFrame, articles: Mapping[tuple[str, str], str],
                   n_correct: int, n_declined: int, seed: int) -> pd.DataFrame:
    """Answers to label: ``n_correct`` the judge called correct, ``n_declined``
    declines, and every unusual one (partly right, or not grounded), since
    those are where a judge is most likely wrong.

    ``articles`` maps (question_id, index) -> the lettered article listing
    the answer was written from.
    """
    rng = np.random.default_rng(seed)
    a = answers.dropna(subset=["correctness", "answer"])
    a = a[a.apply(lambda r: (r["question_id"], r["index"]) in articles, axis=1)]
    unusual = a[(a["correctness"] == 1) | (a["grounded"] == 0)]
    rest = a.drop(unusual.index)
    correct = rest[rest["correctness"] == 2]
    declined = rest[rest["declined"] == 1]
    pick = pd.concat([correct.iloc[rng.permutation(len(correct))[:n_correct]],
                      declined.iloc[rng.permutation(len(declined))[:n_declined]], unusual])
    out = pick.merge(questions[["question_id", "question", "reference_answer"]], on="question_id")
    return pd.DataFrame({"part": "answers", "question": out["question"], "reference_answer": out["reference_answer"],
                         "text": [f"ARTICLES:\n{articles[(q, i)]}\n\nANSWER: {ans}"
                                  for q, i, ans in zip(out["question_id"], out["index"], out["answer"])],
                         "source_id": out["question_id"] + "|" + out["index"],
                         "verdict": [answer_verdict(c, g) for c, g in zip(out["correctness"], out["grounded"])]})


def build_sheet(parts: Sequence[pd.DataFrame], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle every item together; return (sheet for the author, key with verdicts)."""
    items = pd.concat(parts, ignore_index=True)
    items = items.iloc[np.random.default_rng(seed).permutation(len(items))].reset_index(drop=True)
    items["item_id"] = [f"L{i + 1:03d}" for i in range(len(items))]
    items["instructions"] = items["part"].map(INSTRUCTIONS)
    sheet = items.assign(your_label="", your_note="")[SHEET_COLUMNS]
    key = items[["item_id", "part", "source_id", "verdict"]]
    return sheet, key


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Agreement beyond chance between two labellers (1 = perfect, 0 = chance)."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == k).mean() * (b == k).mean() for k in labels))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Agreement between the author's labels and the LLM verdicts, per part.

    Unlabelled rows are skipped and counted. For relevance, "answers" agreement
    (grade 2 vs below) is reported too: that is the cut the metrics rely on.
    """
    m = key.merge(sheet[["item_id", "your_label"]], on="item_id")
    m["your_label"] = pd.to_numeric(m["your_label"], errors="coerce")
    rows = []
    for part, grp in m.groupby("part"):
        done = grp.dropna(subset=["your_label"])
        you, llm = done["your_label"].astype(int).to_numpy(), done["verdict"].astype(int).to_numpy()
        row = {"part": part, "labelled": len(done), "unlabelled": len(grp) - len(done)}
        if len(done):
            row.update(agreement=float((you == llm).mean()), kappa=cohen_kappa(you, llm))
            if part == "relevance":
                row["answers_agreement"] = float(((you == 2) == (llm == 2)).mean())
                row["llm_more_lenient"] = int((llm > you).sum())
                row["llm_stricter"] = int((llm < you).sum())
            else:
                row["llm_kept_you_rejected"] = int(((llm == 1) & (you == 0)).sum())
                row["llm_rejected_you_kept"] = int(((llm == 0) & (you == 1)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_labellers(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Agreement between two people's sheets, per part: the human baseline
    the LLM's agreement is read against. Rows either left blank are skipped."""
    m = a[["item_id", "part", "your_label"]].merge(b[["item_id", "your_label"]], on="item_id", suffixes=("_a", "_b"))
    for c in ("your_label_a", "your_label_b"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["your_label_a", "your_label_b"])
    rows = []
    for part, grp in m.groupby("part"):
        x, y = grp["your_label_a"].astype(int).to_numpy(), grp["your_label_b"].astype(int).to_numpy()
        row = {"part": part, "both_labelled": len(grp), "agreement": float((x == y).mean()), "kappa": cohen_kappa(x, y)}
        if part == "relevance":
            row["answers_agreement"] = float(((x == 2) == (y == 2)).mean())
        rows.append(row)
    return pd.DataFrame(rows)
