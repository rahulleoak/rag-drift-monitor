# Labelling guide

Thank you for helping. This takes about 30–45 minutes.

## What this is for

A computer system answers questions using a collection of news articles
and Wikipedia pages. An AI model checks parts of that system's work. You are
checking the checker: we compare your answers with the AI's to see how
often it gets things right. There are no trick questions, and your honest
judgement is exactly what we need.

## How to fill in the sheet

1. Open `sheet.csv` in Excel. Each row is one item.
2. Look at the **`part`** column first. It tells you which kind of item it
   is: `relevance`, `questions` or `facts`. Each kind has its own rules below.
3. Put your answer (a number) in **`your_label`**.
4. If you are unsure, or something looks odd, write a short reason in
   **`your_note`**. Notes are welcome but optional.
5. Save as CSV when you are done, and send the file back.

**Please:**
- Use only the text in the row. Don't search online or use what you
  happen to know.
- Don't discuss the items with anyone until you have finished.
- If you can't decide, pick the closer answer and add a note.

## Part 1: `relevance` (0, 1 or 2)

**The question:** does this article answer the question?

- The `text` column is the whole article: a headline and a short summary.
  Judge only that text.
- **Ignore the `reference_answer` column.** It came from a different
  article. If it looks wrong, you can say so in a note, but it doesn't
  change your label.

| Label | Meaning |
|---|---|
| **2** | The text answers the question (fully, or the main part of it). |
| **1** | Same subject or event, but the text doesn't give the answer. |
| **0** | A different subject or event, even if some words match. |

Made-up example. Question: *"Why are cake prices rising?"*
- "Cake prices soar as egg and flour costs climb": **2**
- "Bakeries report record cake sales this year": **1** (same subject, no reason given)
- "Bread prices rise after wheat harvest fails": **0** (different subject)

Watch for two similar events, such as two different storms or two
different shootings. If the question is about one and the text is about
the other, it's a **0**.

## Part 2: `questions` (0 or 1)

**The question:** is this a fair question, and is the reference answer
right?

Here, read the `question`, the `reference_answer` and the `text` together.

| Label | Meaning |
|---|---|
| **1** | All three hold: (a) the question is clear and makes sense; (b) the text answers it; (c) the `reference_answer` matches what the text says. |
| **0** | Any of those fails. This includes a reference answer like "not stated in the article": if the article can't answer it, it is not a fair question. |

Made-up example. Text: *"Farm's pig Rosie wins 'prettiest pig' prize."*
- "Which prize did Rosie win?" / "Prettiest pig": **1**
- "How old is Rosie?" / "Prettiest pig": **0** (the text doesn't give her age)

A date in the question ("as of May 2019") comes from when the article
was published. You don't need to find it in the text.

## Part 3: `facts` (0 or 1)

**The question:** between two versions of a Wikipedia sentence, did a fact
change?

The `text` column shows a sentence **BEFORE** an edit and **AFTER** it.
Compare only those two. Neither one is assumed to be correct.

| Label | Meaning |
|---|---|
| **1** | A fact is different: a count, amount, date, ranking or person changed, even by one digit. Also 1 if something added makes the BEFORE sentence wrong or out of date. |
| **0** | The meaning is the same: only the wording, order, rounding, units or number format changed, or new detail was added that leaves BEFORE still true. Also 0 if only an illustrative example changed (e.g. "such as 1/3" → "such as 2/3"). |

Made-up examples:
- "The bridge is 120 m long." → "The bridge is 125 m long.": **1**
- "The town has 8,000 people." → "About 8,000 people live in the town.": **0**
- "The film won 3 awards." → "The film won 3 awards, including Best Actor.": **0**
- "The mayor is Ana Silva (since 2015)." → "The mayor is Ben Okafor (since 2020).": **1**

## Part 4: `answers` (0, 1 or 2), a separate sheet

**The question:** did the assistant answer correctly, using only the
articles it was given?

The `text` column shows the **ARTICLES** the assistant read, then its
**ANSWER**. The `reference_answer` shows the expected answer.

| Label | Meaning |
|---|---|
| **2** | The answer gives the fact asked for correctly, and everything it says is in the ARTICLES. |
| **1** | Partly right (part of the fact missing or slightly off), or it adds something the ARTICLES don't say. |
| **0** | Wrong, or no answer (for example "the archive does not say"), even if declining was the sensible thing to do. |

- "Correct" means it matches the reference answer, or the ARTICLES show it
  is an equally correct answer to the question.
- Check every claim in the answer against the ARTICLES. Something true in
  real life but not in the ARTICLES makes it a **1**.

Made-up example. Question: *"How long is the bridge?"*, reference answer
*"120 metres"*, and the ARTICLES say *"The 120-metre bridge opened in 1932."*
- "It is 120 metres long.": **2**
- "It is 120 metres long and was designed by Ana Silva.": **1** (the designer is not in the ARTICLES)
- "It is 150 metres long.": **0**
- "The articles do not say.": **0**
