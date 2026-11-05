"""All prompt templates in one place, with versions and output schemas.

Changing a prompt's text means bumping its version: the version is part of
the LLM cache key, so old answers are never reused for a new prompt.

Prompts follow the "LLM writes, code decides" rule (D9, D15): the writer
produces drafts, the reviewer answers yes/no checks, and Python decides
what is kept.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------- question writing (Sonnet) ----------

QUESTION_VERSION = "q-v1"

QUESTION_SYSTEM = """You help build a test set for a newsroom archive assistant. Journalists and \
editors ask the assistant questions; it answers from the company's archive of published articles \
(each article is a headline plus a one or two sentence description).

You will be shown one archive article. Write ONE question that a journalist or editor might \
realistically ask the assistant, which this article answers.

Requirements for the question:
- Specific: it should have one clear answer that this article provides (a who, what, where, \
when, how many, or why), not something any article on the topic could answer.
- Natural wording, as a colleague would type it. Do not copy the headline: paraphrase, and avoid \
reusing its distinctive phrases.
- Self-contained: no "this article", "this week", or "the story" without saying what you mean. \
Name the people, places or events involved.
- Between 6 and 30 words.

Requirements for the reference answer:
- One or two sentences, using only what the article says. No outside knowledge.

Also say whether the question is evergreen (its answer stays true over time, e.g. what someone \
did or said on a given occasion) or time_sensitive (its answer could change, e.g. current \
guidance, who holds an office now, latest figures).

If the article states no answerable fact (pure opinion, a teaser, a list with no content), still \
write your best question but set answerable_from_article to false."""

QUESTION_TEMPLATE = """Article (published {date}, section {category}):
Headline: {title}
Description: {text}"""


class QuestionDraft(BaseModel):
    """What the question writer returns for one article."""

    question: str = Field(description="The question a journalist or editor would ask")
    reference_answer: str = Field(description="One or two sentences, from the article only")
    question_type: Literal["evergreen", "time_sensitive"]
    answerable_from_article: bool


# ---------- question review (Opus) ----------

REVIEW_VERSION = "r-v1"

REVIEW_SYSTEM = """You check questions for a test set of a newsroom archive assistant. Each \
question was written from one archive article (a headline plus a short description) and comes \
with a reference answer. Judge strictly and only from the article text shown; do not use outside \
knowledge to fill gaps.

Answer each check with true or false:
- answerable: the article alone fully answers the question.
- unambiguous: the question has one clear answer and names what it is about (no unexplained \
"this", "the story", "this week").
- needs_article: answering it needs this article's information; it is not common knowledge or \
answerable from the question itself.
- reference_correct: the reference answer is correct and supported by the article.
- natural: a journalist or editor could plausibly ask it this way (not a copy of the headline, \
not a quiz-style trick question).

Then give a reason of at most 25 words, naming the main problem if any check is false."""

REVIEW_TEMPLATE = """Article (published {date}, section {category}):
Headline: {title}
Description: {text}

Question: {question}
Reference answer: {reference_answer}"""


class QuestionReview(BaseModel):
    """The reviewer's checks; code decides whether the question is kept."""

    answerable: bool
    unambiguous: bool
    needs_article: bool
    reference_correct: bool
    natural: bool
    reason: str = Field(description="At most 25 words")

    def passes(self) -> bool:
        return all((self.answerable, self.unambiguous, self.needs_article,
                    self.reference_correct, self.natural))
