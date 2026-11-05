"""All prompt templates in one place, with versions and output schemas.

Changing a prompt's text means bumping its version: the version is part of
the LLM cache key, so old answers are never reused for a new prompt.

Prompts follow the "LLM writes, code decides" rule (D9, D15): the writer
produces drafts, the reviewer answers yes/no checks, and Python decides
what is kept.

Version history:
    q-v1 / r-v1  first pilot (F7)
    q-v2 / r-v2  one fact per question; up to two questions per article (one
                 evergreen, one time-sensitive); direct reference answers; no
                 outside facts; sharper time-sensitive rule; reviewer checks
                 for single question, outside facts and the type label, and
                 must answer false whenever its reason names a problem (D17)
    q-v3 / r-v3  the headline counts as article content; common knowledge about
                 who people and places are is allowed, invented event details are
                 not; reviewer check "no_outside_facts" becomes
                 "no_invented_details" (D17, revised after the second pilot)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------- question writing (Sonnet) ----------

QUESTION_VERSION = "q-v3"

QUESTION_SYSTEM = """You help build a test set for a newsroom archive assistant. Journalists and \
editors ask the assistant questions; it answers from the company's archive of published articles \
(each article is a headline plus a one or two sentence description).

You will be shown one archive article. The headline and the description are both part of the \
article: a fact stated only in the headline counts. Write questions that a journalist or editor \
might realistically ask the assistant, which this article answers.

Write one or two questions:
- Always one question about a fact the article states.
- Add a second question only if the article also states a time_sensitive fact (see below) that \
your first question does not already ask about. Never write two questions about the same fact.

Each question must:
- Ask for exactly ONE fact: one who, what, where, when, how many, or why. Never join two asks \
with "and".
- Be specific: one clear answer that this article gives, not something any article on the topic \
could answer.
- Do not invent details about the event: no dates, places, numbers, causes or outcomes that the \
article does not state (the publication date may be used). Common knowledge about who people and \
places are is fine, as a journalist would naturally write it (e.g. "Canadian Prime Minister \
Justin Trudeau", "Pennsylvania senator Pat Toomey").
- Use natural wording, as a colleague would type it. Paraphrase: do not reuse distinctive \
phrases from the headline or the description.
- Be self-contained: no "this article", "this week", or "the story"; name the people, places or \
events involved.
- Be between 6 and 30 words.

Each reference answer must:
- State the fact directly in one or two sentences, from the article only.
- Never describe the article ("The article says...", "The description...").

Question type:
- time_sensitive: the answer could change or be revised later. This includes counts and tolls \
that were still being reported ("so far", "at least"), current guidance or rules, who holds an \
office or role now, prices, rankings, and anything described as ongoing.
- evergreen: the answer stays true, such as what someone did or said on a given occasion, or a \
final result.

If the article states no answerable fact (pure opinion, a teaser that promises information without \
giving it, a list with no content), write the one question a journalist would most naturally ask \
about its topic anyway, set answerable_from_article to false, and set the reference answer to \
"Not stated in the article.\""""

QUESTION_TEMPLATE = """Article (published {date}, section {category}):
Headline: {title}
Description: {text}"""


class DraftQuestion(BaseModel):
    """One question with its reference answer."""

    question: str = Field(description="One question asking for exactly one fact")
    reference_answer: str = Field(description="The fact stated directly, from the article only")
    question_type: Literal["evergreen", "time_sensitive"]


class QuestionDraft(BaseModel):
    """What the question writer returns for one article."""

    questions: list[DraftQuestion] = Field(description="One or two questions")
    answerable_from_article: bool


# ---------- question review (Opus) ----------

REVIEW_VERSION = "r-v3"

REVIEW_SYSTEM = """You check questions for a test set of a newsroom archive assistant. Each \
question was written from one archive article (a headline plus a short description) and comes \
with a reference answer and a type label. The headline and the description are both part of the \
article. Judge whether the question is answered only from the article text shown.

Answer each check with true or false:
- answerable: the article alone fully answers the question.
- unambiguous: the question has one clear answer and names what it is about (no unexplained \
"this", "the story", "this week").
- single_question: it asks for exactly one fact (not two asks joined by "and", not a list).
- no_invented_details: the question invents no details about the event (dates, places, numbers, \
causes, outcomes) that the article does not state. The publication date may be used. Common \
knowledge about who people and places are (a politician's country or office, what a well-known \
company does) is fine and is NOT a problem.
- needs_article: answering it needs this article's information; it is not common knowledge or \
answerable from the question itself.
- reference_correct: the reference answer is correct, supported by the article, and states the \
fact directly.
- natural: a journalist or editor could plausibly ask it this way, without copying the headline \
or description wording.
- type_correct: the type label is right. time_sensitive means the answer could change or be \
revised later (counts still being reported, current rules, who holds a role now, ongoing \
situations); evergreen means it stays true.

Be strict: if you notice any problem with a check, even a minor one, that check is false. Your \
reason must not describe a problem while the matching check is true.

Then give a reason of at most 25 words, naming the main problem if any check is false."""

REVIEW_TEMPLATE = """Article (published {date}, section {category}):
Headline: {title}
Description: {text}

Question: {question}
Reference answer: {reference_answer}
Type label: {question_type}"""

REVIEW_CHECKS = ("answerable", "unambiguous", "single_question", "no_invented_details",
                 "needs_article", "reference_correct", "natural", "type_correct")


class QuestionReview(BaseModel):
    """The reviewer's checks; code decides whether the question is kept."""

    answerable: bool
    unambiguous: bool
    single_question: bool
    no_invented_details: bool
    needs_article: bool
    reference_correct: bool
    natural: bool
    type_correct: bool
    reason: str = Field(description="At most 25 words")
