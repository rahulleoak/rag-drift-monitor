"""Tests for the changed-facts check (D27)."""
from driftmon.drift import facts

OLD = [("Lead", "The Black Death killed over 75 million people in 1350. It spread along trade routes."),
       ("Cases", "There were 2217 known species. The museum opened in 1999 in Kyiv.")]


def test_changed_number_is_found_with_evidence():
    new = [("Lead", "The Black Death killed over 20 million people in 1350. It spread along trade routes."),
           ("Cases", "There were 2217 known species. The museum opened in 1999 in Kyiv.")]
    got = facts.fact_changes(OLD, new)
    assert len(got) == 1
    c = got[0]
    assert c.section == "Lead" and c.removed == ("75",) and c.added == ("20",)
    assert "75 million" in c.before and "20 million" in c.after


def test_formatting_rewording_and_new_statements_are_not_facts():
    new = [("Lead", "The Black Death killed over 75 million people in 1350. It spread quickly along the old "
                    "trade routes of the time."),                                   # reworded, same numbers
           ("Cases", "There were 2,217 known species. The museum opened in 1999 in Kyiv. "  # formatting only
                     "A new stadium seating 40,000 was built in 2012.")]           # new statement
    assert facts.fact_changes(OLD, new) == []


def test_numbers_are_normalised():
    assert facts.numbers("Over 2,217 species, up from 1,000.") == ["2217", "1000"]


def test_added_or_removed_numbers_citations_and_leftovers_are_not_facts():
    old = [("Lead", "A study examined over 464,000 recordings of popular music made between 1955 and 2010. "
                    "Jerry Mander, In the Absence of the Sacred, Sierra Club Books, 1992, ISBN 0-87156-509-9. "
                    "Mao Zedong (December 26, 1893 to September 9, 1976) was a Chinese revolutionary leader.")]
    new = [("Lead", "In 2016, a study examined over 464,000 recordings of popular music made between 1955 and 2010. "
                    "Jerry Mander, In the Absence of the Sacred, Sierra Club Books, 1992, ISBN 0-87156-509-1. "
                    "Mao Zedong (26 December 18939 September 1976) was a Chinese revolutionary leader.")]
    assert facts.fact_changes(old, new) == []


def test_image_captions_are_not_facts():
    old = [("Lead", "right|thumb|150px|Bust of Socrates in the Vatican Museum in Rome, Italy.")]
    new = [("Lead", "right|thumb|upright=0.7|Bust of Socrates in the Vatican Museum in Rome, Italy.")]
    assert facts.fact_changes(old, new) == []


def test_wording_edits_are_the_comparison_group():
    old = [("Lead", "The bridge, built in 1932, is 120 metres long and crosses the river. It is painted red.")]
    new = [("Lead", "The bridge, completed in 1932, is 120 metres long and spans the river. It is painted red.")]
    got = facts.wording_edits(old, new)
    assert len(got) == 1 and "completed" in got[0][2] and facts.fact_changes(old, new) == []
    changed = [("Lead", "The bridge, built in 1932, is 125 metres long and crosses the river. It is painted red.")]
    assert facts.wording_edits(old, changed) == [] and len(facts.fact_changes(old, changed)) == 1
