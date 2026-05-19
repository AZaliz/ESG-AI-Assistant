"""Task 2: make_label heuristics for wide-header tables.

Run: python tests/test_make_label.py   (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # esg_scraper/

from parser.table_normalizer import make_label


def test_standard_label_in_col0():
    row = ["Total energy consumption", "MWh", "3,703,856"]
    assert make_label(row) == "Total energy consumption"


def test_esrs_code_promotes_col1_and_keeps_ref():
    row = ["ESRS E1-6", "Gross Scope 1 GHG emissions", "ktCO2e", "451"]
    assert make_label(row) == "ESRS E1-6 | Gross Scope 1 GHG emissions"


def test_garbage_label_dropped():
    row = ["", "", "column 4: 156,921"]
    assert make_label(row) is None


def test_extra_cases():
    # GRI code with real KPI in col1 -> concat
    assert make_label(["GRI 305", "Direct GHG emissions", "t", "10"]) == \
        "GRI 305 | Direct GHG emissions"
    # tiny junk col0, real label col1
    assert make_label(["-", "Water withdrawal", "m3", "12"]) == "Water withdrawal"
    # plain "column N" label -> drop
    assert make_label(["column 2", "1,234"]) is None
    # short noise -> drop
    assert make_label(["Tot", "9"]) is None
    # code but col1 numeric -> falls back to col1 (numeric) -> too short -> drop
    assert make_label(["SASB", "451"]) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
