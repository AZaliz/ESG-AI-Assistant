"""Schema-tagging classifiers: speaker_role + esrs_topic (parser/taxonomy.py).

These back the durability fix — the build pipeline now emits speaker_role
and esrs_topic natively (no _schema_patch step). Tests assert the ACTUAL
function signatures: speaker_role_for_turn() takes the transcript turn's
role token (the earnings ingester already has it split out), not chunk text.

Run: python tests/test_taxonomy.py   (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # esg_scraper/

from parser.taxonomy import classify_esrs_topics, speaker_role_for_turn


def test_speaker_role_maps_known_tokens():
    assert speaker_role_for_turn("Executives") == "executive"
    assert speaker_role_for_turn("Analysts") == "analyst"
    assert speaker_role_for_turn("Operator") == "operator"
    assert speaker_role_for_turn("Attendees") == "management"


def test_speaker_role_unknown_is_none():
    # defensive: unrecognized token / empty / None -> None
    assert speaker_role_for_turn("UnknownRole") is None
    assert speaker_role_for_turn("") is None
    assert speaker_role_for_turn(None) is None


def test_esrs_e1_climate():
    chunk_e1 = "Our Scope 1 GHG emissions decreased to 38 MtCO2e in 2024..."
    assert "E1" in classify_esrs_topics(chunk_e1)


def test_esrs_s1_own_workforce():
    chunk_s1 = "The gender pay gap in our own workforce stands at 1.8%..."
    assert "S1" in classify_esrs_topics(chunk_s1)


def test_esrs_no_topic_is_empty_list():
    chunk_neither = "Financial highlights for Q1 included revenue growth..."
    result = classify_esrs_topics(chunk_neither)
    assert result == []
    assert isinstance(result, list)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
