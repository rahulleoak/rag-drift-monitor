"""Tests for chunking and Wikipedia cleaning (D26). No network."""
from driftmon import chunking, wikipedia


def words(text: str) -> int:
    return len(text.split())


def test_chunks_respect_size_sections_and_overlap():
    para = " ".join(f"Sentence number {i} is here." for i in range(40))   # 200 words
    secs = [("Lead", para), ("History", "Short section with enough words to stand alone here.")]
    chunks = chunking.chunk_document(secs, words, size=50, overlap=10, min_tokens=5)
    assert {c.section for c in chunks} == {"Lead", "History"}
    lead = [c for c in chunks if c.section == "Lead"]
    assert len(lead) >= 4 and all(c.tokens <= 50 + 10 for c in lead)
    # each chunk after the first starts with the last sentence(s) of the previous one
    first_of_next = lead[1].text.split(". ")[0] + "."
    assert lead[0].text.endswith(lead[0].text.split(first_of_next)[-1]) and first_of_next in lead[0].text
    assert lead[0].text.split(". ")[-1] in lead[1].text
    assert chunks[-1].section == "History"            # never merged across a section boundary


def test_tiny_final_piece_is_merged():
    text = "One two three four five six seven eight nine ten.\n\nTail bit."
    chunks = chunking.chunk_section("S", text, words, size=10, overlap=0, min_tokens=5)
    assert len(chunks) == 1 and chunks[0].text.endswith("Tail bit.")


def test_truncate_keeps_whole_words_within_limit():
    text = " ".join(f"w{i}" for i in range(100))
    out = chunking.truncate(text, words, limit=30)
    assert words(out) == 30 and out.endswith("w29")


WIKITEXT = """{{Infobox person|name=X}}'''Plato''' ({{lang|grc|Πλάτων}}; 428 BC) was a [[philosopher|Greek philosopher]] in Athens.<ref>Source.</ref>
He founded the Academy, a school of philosophy.
[[File:Plato.jpg|thumb|A bust of Plato]]

== Biography ==
=== Early life ===
Plato was born into an aristocratic family in Athens.
{| class="wikitable"
| cell || cell
|}

== References ==
{{reflist}}
* A book that should not appear in the text.
[[Category:Philosophers]]
"""


def test_sections_are_clean_and_reference_sections_dropped():
    secs = wikipedia.sections(WIKITEXT, ["references"])
    assert [h for h, _ in secs] == ["Lead", "Biography / Early life"]
    lead = secs[0][1]
    assert "Greek philosopher" in lead and "Infobox" not in lead and "Source." not in lead
    assert "thumb" not in lead and "bust" not in lead and "(;" not in lead and "( ;" not in lead
    assert "cell" not in secs[1][1] and "should not appear" not in " ".join(t for _, t in secs)


def test_vital_titles_and_stratified_sample():
    page = "Intro [[Not a list item]]\n==People==\n* {{Icon|B}} [[Plato]]\n* [[Homer|Homer (poet)]]\n" \
           "==Science==\n* [[Physics]]\n* [[Chemistry]]\n* [[Biology]]\n"
    by = wikipedia.vital_titles(page)
    assert by == {"People": ["Plato", "Homer"], "Science": ["Physics", "Chemistry", "Biology"]}
    picks = wikipedia.sample_titles(by, per_section=2, seed=0)
    assert len(picks) == 4 and sum(s == "Science" for s, _ in picks) == 2
    assert picks == wikipedia.sample_titles(by, per_section=2, seed=0)     # reproducible
    assert wikipedia.is_redirect("#REDIRECT [[Plato]]")
