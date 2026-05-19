"""Task 1 regression: narrative_only must suppress all table extraction.

Runs against the real CSDDD PDF (first pages are enough). With the default
parser the EUR-Lex two-column layout produces phantom tables; narrative_only
must yield zero tables and non-empty text.

Run: python -m pytest tests/test_parser_narrative_only.py -q
  or: python tests/test_parser_narrative_only.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # esg_scraper/

from parser.parser import parse_pdf_to_list

CSDDD = Path("data/pdfs/eu_regulators/2024_regulation_64f32378.pdf")


def test_narrative_only_suppresses_tables():
    assert CSDDD.exists(), f"fixture missing: {CSDDD}"

    default = parse_pdf_to_list(CSDDD, max_pages=12)
    narrative = parse_pdf_to_list(CSDDD, max_pages=12, narrative_only=True)

    default_tables = sum(len(p.tables) for p in default)
    narrative_tables = sum(len(p.tables) for p in narrative)

    # The bug: default parsing finds phantom tables in the OJ column layout.
    assert default_tables > 0, "expected phantom tables under default parsing"
    # The fix: narrative_only finds none.
    assert narrative_tables == 0, f"narrative_only leaked {narrative_tables} tables"
    # And still extracts prose.
    assert any(p.text.strip() for p in narrative), "narrative_only produced no text"


if __name__ == "__main__":
    test_narrative_only_suppresses_tables()
    print("PASS test_narrative_only_suppresses_tables")
