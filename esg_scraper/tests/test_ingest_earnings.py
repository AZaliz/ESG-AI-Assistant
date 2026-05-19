"""Task 4: earnings transcript header-strip + year extraction.

Run: python tests/test_ingest_earnings.py   (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # esg_scraper/

from ingest_earnings import _strip_header, _year, _speaker_chunks

_DM = {
    "company": "Test Co",
    "doc_type": "earnings_call",
    "fiscal_year": 2024,
    "publication_year": 2025,
    "source_file": "Test_Co,_2024_Earnings_Call,_Jan_01,_2025.txt",
    "source_authority": 2,
}

SAMPLE = """Title: Airbus SE, 2022 Earnings Call, Feb 16, 2023
Event Type: Earnings Calls
Event Date: 2023-02-16T06:30:00Z
Source: Koyfin (API)
================================================================================

Operator  [Operator]
Ladies and gentlemen, welcome.
"""


def test_strip_header_removes_metadata_block():
    body = _strip_header(SAMPLE)
    assert body.startswith("Operator")
    assert "Event Type:" not in body
    assert "Koyfin" not in body


def test_strip_header_no_divider_keeps_text():
    assert _strip_header("just body, no divider") == "just body, no divider"


def test_year_prefers_period_then_falls_back():
    assert _year("FY 2022", "file_2099.txt", "Feb 16, 2023") == 2022
    assert _year("Nine Months 2024", "x") == 2024
    import math
    assert _year(float("nan"), "Airbus,_FY_2021_Call.txt") == 2021   # NA period
    assert _year(None, None) is None


def test_operator_turn_skipped_executive_kept():
    # One Operator turn then one Executive turn -> only the executive chunk
    # is emitted; the operator turn still advances the counter so the
    # surviving chunk_id keeps its pre-filter number (n0002, not n0001).
    body = (
        "Operator  [Operator]\n"
        "Good morning. Your next question comes from the line of Jane Doe.\n"
        "Jane Doe  [Executives]\n"
        "We delivered strong results this quarter with solid margins.\n"
    )
    chunks = _speaker_chunks(body, _DM)
    assert len(chunks) == 1
    assert chunks[0].speaker_role == "executive"
    assert chunks[0].chunk_id.endswith("_n0002")  # operator was n0001, skipped


def test_only_operator_turns_yields_zero_chunks():
    body = (
        "Operator  [Operator]\n"
        "Welcome to the call. [Operator Instructions]\n"
        "Operator  [Operator]\n"
        "That concludes today's conference. Thank you.\n"
    )
    assert _speaker_chunks(body, _DM) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
