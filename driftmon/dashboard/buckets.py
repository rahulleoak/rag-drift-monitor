"""The four kinds of drift, as the dashboard names them (D38).

One score can drop for different reasons, and each reason has its own fix:

    knowledge   the archive holds outdated content, or lacks the new content
    query       people ask in new words; the documents are fine
    retrieval   the right content is in the index but is not found or ranked
    producing   the right articles reach the model and the answer is still wrong

The dashboard puts a bucket's name only where that bucket is measured, so a
label never claims more than the number beside it. The wording lives here,
once: the cause column on Impact, the chips on Recommendations and the hover
notes all read from these tables. Display only: the stored causes and actions
that Agent E's rules read are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from driftmon.evaluation.attribution import KNOWLEDGE, PRODUCING, RETRIEVAL
from driftmon.policy import (ANSWER_STEP, CHUNKING, ESCALATE, MISSING_CONTENT, MODEL_EVAL, MONITOR, PROMPTS,
                             REINDEX, REINDEX_DOCS, RETRAIN, RETRIEVAL_CONFIG, RETRIEVAL_OR_MODEL)

QUERY = "query"


@dataclass(frozen=True)
class Bucket:
    name: str        # shown: "Knowledge"
    means: str       # what went wrong
    measured: str    # how this system measures it
    fix: str         # the kind of action that fixes it

    def note(self) -> str:
        return f"{self.name}: {self.means} Measured by {self.measured}. Fix: {self.fix}."


BUCKETS = {
    KNOWLEDGE: Bucket("Knowledge", "the archive holds outdated content, or lacks the new content.",
                      "stale vs refreshed answers and the changed-facts check",
                      "update the documents and re-index"),
    QUERY: Bucket("Query", "people ask in new words; the documents are fine.",
                  "drift on the questions themselves and the newer-wording test",
                  "query rewriting, synonyms or prompt changes"),
    RETRIEVAL: Bucket("Retrieval", "the right content is in the index but is not found or ranked high enough.",
                      "search quality on the refreshed index (the content is there)",
                      "chunking, the embedding model or retrieval settings"),
    PRODUCING: Bucket("Producing", "the right articles reach the model and the answer is still wrong.",
                      "answers judged wrong although the top 5 held an answer",
                      "the answer prompt, or evaluating the answer model"),
}
ORDER = (KNOWLEDGE, QUERY, RETRIEVAL, PRODUCING)
ALL = "All kinds"

# Agent D's likely cause per section -> its bucket
CAUSE = {MISSING_CONTENT: KNOWLEDGE, RETRIEVAL_OR_MODEL: RETRIEVAL, ANSWER_STEP: PRODUCING}

# Agent E's actions -> the bucket(s) they fix; () = every kind (monitoring, a person's judgement).
# Model evaluation and retraining are about the *embedding* model here: their rule fires on poor
# retrieval (policy.model_evaluation), so they fix retrieval, not the answer model.
ACTION = {REINDEX: (KNOWLEDGE,), REINDEX_DOCS: (KNOWLEDGE,), CHUNKING: (RETRIEVAL,),
          RETRIEVAL_CONFIG: (RETRIEVAL,), PROMPTS: (QUERY, PRODUCING), MODEL_EVAL: (RETRIEVAL,),
          RETRAIN: (RETRIEVAL,), ESCALATE: (), MONITOR: ()}


def legend(kinds: tuple[str, ...] = ORDER) -> str:
    """One line per bucket, for a hover note."""
    return "  \n".join(BUCKETS[k].note() for k in kinds)


def cause_label(cause: str | None) -> str:
    """'missing content (re-index)' -> 'Knowledge · missing content (re-index)'; other text unchanged."""
    if not cause or not isinstance(cause, str):
        return ""
    kind = CAUSE.get(cause)
    return f"{BUCKETS[kind].name} · {cause}" if kind else cause


def action_kinds(key: str) -> str:
    """The chip on a recommendation: 'Knowledge', 'Query · Producing' or 'All kinds'."""
    kinds = ACTION.get(key)
    if kinds is None:
        return ""
    return " · ".join(BUCKETS[k].name for k in kinds) if kinds else ALL


def action_note(key: str) -> str:
    """The hover note for a recommendation's chip."""
    kinds = ACTION.get(key)
    if kinds is None:
        return ""
    if not kinds:
        return ("Covers every kind of drift (knowledge, query, retrieval, producing): it keeps watching, or "
                "hands a case to a person.")
    return legend(kinds)
