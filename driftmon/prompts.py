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
    r-v4         the reviewer assigns the evergreen / time-sensitive label itself
                 (without seeing the writer's) instead of grading it, because the
                 label proved ambiguous; a disagreement is recorded, not a
                 rejection (D17, revised after the third pilot). Writer unchanged.
    r-v5         "natural" no longer counts reused names, places, numbers or facts
                 as copying; only awkward phrasing or a whole copied sentence fails
                 it. Measured reason: copy-rejected questions were not easier to
                 retrieve (F11). Verbatim copying is caught in code. Writer unchanged.
    rel-v1       retrieval relevance judge (Agent C): grades 0/1/2 for the pooled
                 top-k of both indexes, in shuffled order, index not shown (D25)
    ans-v1       RAG answers from the retrieved top-k only (Agent D, D28)
    aj-v1        answer judge: correctness 0/1/2, grounded, declined; one call per
                 answer, index not shown (D28)
    fq-v1 / wans-v1 / ac-v1  outdated-answer test on edited Wikipedia passages:
                 a question from the new sentence, answers from each version's
                 passages, and whether the two answers differ (D27, D28)
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

REVIEW_VERSION = "r-v5"

REVIEW_SYSTEM = """You check questions for a test set of a newsroom archive assistant. Each \
question was written from one archive article (a headline plus a short description) and comes \
with a reference answer. The headline and the description are both part of the article. Judge \
whether the question is answered only from the article text shown.

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
- natural: a journalist or editor could plausibly ask it this way: clear, idiomatic phrasing. \
Reusing the article's names, places, numbers or facts is normal and is NOT a problem. Mark it \
false only for awkward or unnatural phrasing (for example a question that half-answers itself), \
or when a whole sentence of the article is copied as the question.
Be strict: if you notice any problem with a check, even a minor one, that check is false. Your \
reason must not describe a problem while the matching check is true.

Also label the question (this is a classification, not a pass/fail check):
- time_sensitive: the correct answer could change or be revised after publication (counts or \
tolls still being reported, current rules or guidance, who holds a role now, ongoing situations).
- evergreen: the correct answer stays true (what someone did or said on a given occasion, a final \
result, a fixed fact).

Then give a reason of at most 25 words, naming the main problem if any check is false."""

REVIEW_TEMPLATE = """Article (published {date}, section {category}):
Headline: {title}
Description: {text}

Question: {question}
Reference answer: {reference_answer}"""

REVIEW_CHECKS = ("answerable", "unambiguous", "single_question", "no_invented_details",
                 "needs_article", "reference_correct", "natural")


class QuestionReview(BaseModel):
    """The reviewer's checks; code decides whether the question is kept."""

    answerable: bool
    unambiguous: bool
    single_question: bool
    no_invented_details: bool
    needs_article: bool
    reference_correct: bool
    natural: bool
    question_type: Literal["evergreen", "time_sensitive"]
    reason: str = Field(description="At most 25 words")


# ---------- retrieval relevance (Opus, Agent C) ----------

RELEVANCE_VERSION = "rel-v1"

RELEVANCE_SYSTEM = """You judge search results for a newsroom archive assistant. Journalists and \
editors ask it questions; it searches the company's archive of published articles (each article is \
a headline plus a short description) and answers from the articles it finds.

You will see one question, a reference answer, and a list of archive articles in random order. The \
reference answer was written from one article; other articles may answer the question equally \
well, and some articles may not. Grade every article on its own, from its text only:
- 2: the article answers the question: it states the fact asked for (or an equally correct answer \
to the question as asked). If the question is about a specific event, person, place or time, the \
article must be about that same one.
- 1: on the same topic or event and useful background, but it does not state the answer.
- 0: not relevant: a different topic, or a different event that only shares words with the \
question.

Do not use outside knowledge to fill gaps: judge what each article itself says. Grade every \
article listed, exactly once, using its letter."""

RELEVANCE_TEMPLATE = """Question: {question}
Reference answer: {reference_answer}

Articles:
{articles}"""


class ArticleGrade(BaseModel):
    """One article's relevance grade."""

    article: str = Field(description="The article's letter")
    grade: Literal[0, 1, 2]


class RelevanceJudgement(BaseModel):
    """Grades for every pooled article of one question; code computes the metrics."""

    grades: list[ArticleGrade]


# ---------- RAG answers (Sonnet, Agent D) ----------

ANSWER_VERSION = "ans-v1"

ANSWER_SYSTEM = """You are a newsroom archive assistant. Journalists and editors ask you questions; \
you answer only from the archive articles provided with each question (each article is a headline \
plus a short description, with its publication date).

Rules:
- Use only what the articles say. Do not add facts from your own knowledge, even if you know them.
- If the articles do not contain the answer, say so plainly and set "found" to false. Do not guess.
- If a question asks about a specific event, person, place or time, answer only from articles about \
that same one.
- Keep the answer short and direct: one to three sentences.
- List the letters of the articles your answer relies on."""

ANSWER_TEMPLATE = """Question: {question}

Archive articles (most relevant first):
{articles}"""


class RagAnswer(BaseModel):
    """The assistant's answer, from the retrieved articles only."""

    found: bool = Field(description="True if the articles contain the answer")
    answer: str = Field(description="One to three sentences; if not found, say the archive does not say")
    sources: list[str] = Field(description="Letters of the articles the answer relies on")


# ---------- answer quality (Opus, Agent D) ----------

ANSWER_JUDGE_VERSION = "aj-v1"

ANSWER_JUDGE_SYSTEM = """You check answers from a newsroom archive assistant. It answers questions \
using only the archive articles it retrieved (each a headline plus a short description, with its \
publication date).

You will see a question, a reference answer written from one article, the articles the assistant \
retrieved, and the assistant's answer. Judge the answer on three points:
- correctness: 2 if it gives the fact asked for, matching the reference answer or an answer the \
retrieved articles show is equally correct for the question as asked; 1 if partly right (a key part \
missing or slightly off); 0 if wrong, or if it gives no answer.
- grounded: true if every factual claim in the answer is stated in the retrieved articles; false if \
any claim is not (invented, or from outside knowledge). An answer that only says the articles do \
not contain the answer is grounded.
- declined: true if the answer says the articles do not contain the answer instead of answering.

Judge from the texts shown only; do not use outside knowledge to decide what is true."""

ANSWER_JUDGE_TEMPLATE = """Question: {question}
Reference answer: {reference_answer}

Retrieved articles:
{articles}

Assistant's answer: {answer}"""


class AnswerJudgement(BaseModel):
    """Grades for one answer; code turns them into metrics and decisions."""

    reason: str = Field(description="At most 30 words")
    correctness: Literal[0, 1, 2]
    grounded: bool
    declined: bool


# ---------- outdated-answer test on edited documents (Agent D, D27 / D28) ----------

FACT_QUESTION_VERSION = "fq-v1"

FACT_QUESTION_SYSTEM = """You help build a test set for an assistant that answers questions from \
an encyclopedia. You will see one sentence from an encyclopedia article, with the article's title \
and section.

Write one question that a reader could ask, whose answer is a specific fact stated in the sentence: \
a number, amount, date, count, ranking or name. Rules:
- The question must make sense on its own: name the subject (do not say "the article" or "this \
sentence").
- Ask for one fact only, and do not put the answer in the question.
- Give the answer as stated in the sentence, in a few words."""

FACT_QUESTION_TEMPLATE = """Article: {title}
Section: {section}
Sentence: {sentence}"""


class FactQuestion(BaseModel):
    """One question about a fact stated in a sentence."""

    question: str
    answer: str = Field(description="A few words, as stated in the sentence")


WIKI_ANSWER_VERSION = "wans-v1"

WIKI_ANSWER_SYSTEM = """You are a reference assistant. You answer questions only from the \
encyclopedia passages provided with each question.

Rules:
- Use only what the passages say. Do not add facts from your own knowledge, even if you know them, \
and even if you believe a passage is out of date.
- If the passages do not contain the answer, say so plainly and set "found" to false. Do not guess.
- Keep the answer short and direct: one or two sentences, giving the specific fact asked for.
- List the letters of the passages your answer relies on."""

WIKI_ANSWER_TEMPLATE = """Question: {question}

Passages (most relevant first):
{passages}"""


ANSWER_CHANGE_VERSION = "ac-v1"

ANSWER_CHANGE_SYSTEM = """You compare two answers to the same question, given by an assistant \
working from two different versions of a reference archive. Decide whether they state the same fact.

- same: both give the same fact (wording, rounding or format may differ, e.g. "2,217" and "2217", \
"about 8,000" and "8,000").
- different: both give an answer, and the facts differ (a different number, date, amount, ranking \
or name), even slightly.
- one_declined: one answer gives the fact and the other says the information is not available.
- both_declined: neither gives the fact.

Judge only whether the two answers agree; do not decide which one is true."""

ANSWER_CHANGE_TEMPLATE = """Question: {question}

Answer 1: {answer_1}

Answer 2: {answer_2}"""


class AnswerChange(BaseModel):
    """Whether two answers to one question state the same fact."""

    reason: str = Field(description="At most 25 words")
    verdict: Literal["same", "different", "one_declined", "both_declined"]
