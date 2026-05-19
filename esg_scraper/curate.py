"""Curate the scraper manifest down to a RAG-ready subset.

The full manifest (~257 docs / 2.7 GB) is great for archival, but most of it
is wrong shape for RAG: xlsx databooks, duplicate hashes, pre-2022 reports
that would pollute retrieval, one-pagers, and 100+ MB monster PDFs.

This script applies a defensible default filter and writes
`data/curated_manifest.csv` that downstream tools (integrate.py → rag-build)
can consume safely.

Defaults (override with flags):
  - Drop file_format != pdf (xlsx databooks aren't text)
  - Drop duplicate file_hash (keep first occurrence)
  - Drop priority 3 (archive-only)
  - Drop fiscal_year < 2023 (recent disclosures only)
  - Drop size < 500 KB (one-pagers, content indices)
  - Drop size > 80 MB (cost-prohibitive to embed)
  - Drop missing file_hash / size_bytes / file_path

Usage:
    python curate.py                              # data/manifest.csv → data/curated_manifest.csv
    python curate.py --include-secondary          # keep priority 2 docs too
    python curate.py --min-year 2022              # widen historical window
    python curate.py --max-size-mb 120            # allow larger PDFs
    python curate.py --include-xlsx               # keep databooks
"""
# --- UTF-8 console bootstrap: Windows wraps stdout in cp1252, which can't
# encode →, —, É (L'Oréal), etc. Force UTF-8 so output never crashes. ---
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "curated_manifest.csv"


def banner(s: str):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


def apply_filters(df: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    """Apply the curation filters. Returns (filtered_df, drop_stats)."""
    stats = {"start": len(df)}
    n = len(df)

    # 1. Drop rows missing critical fields
    before = n
    df = df.dropna(subset=["file_path", "file_hash", "size_bytes"])
    stats["dropped_missing_fields"] = before - len(df)
    n = len(df)

    # 2. Drop non-PDF unless --include-xlsx
    if not args.include_xlsx:
        before = n
        df = df[df["file_format"].str.lower() == "pdf"]
        stats["dropped_non_pdf"] = before - len(df)
        n = len(df)

    # 3. Drop duplicate file_hash (keep first — typically the higher-priority one)
    before = n
    df = df.sort_values("priority").drop_duplicates(subset="file_hash", keep="first")
    stats["dropped_duplicate_hash"] = before - len(df)
    n = len(df)

    # 4. Drop priority 3 unless flag — and priority 2 unless --include-secondary
    if args.include_secondary:
        before = n
        df = df[df["priority"] <= 2]
        stats["dropped_priority_3_only"] = before - len(df)
    else:
        before = n
        df = df[df["priority"] == 1]
        stats["dropped_priority_2_or_3"] = before - len(df)
    n = len(df)

    # 5. Fiscal year filter (NaN years are kept — undated but possibly recent)
    before = n
    df = df[df["fiscal_year"].isna() | (df["fiscal_year"] >= args.min_year)]
    stats["dropped_too_old"] = before - len(df)
    n = len(df)

    # 6. Size filters
    before = n
    df = df[df["size_bytes"] >= args.min_size_kb * 1000]
    stats["dropped_too_small"] = before - len(df)
    n = len(df)

    before = n
    df = df[df["size_bytes"] <= args.max_size_mb * 1_000_000]
    stats["dropped_too_large"] = before - len(df)
    n = len(df)

    stats["kept"] = len(df)
    return df.reset_index(drop=True), stats


def absolutize_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Make `file_path` absolute relative to the script's project root.

    The scraper writes paths like `data\\pdfs\\...` (relative to its own dir).
    Downstream tools (integrate.py) work better with absolute paths.
    """
    def _abs(p):
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((SCRIPT_DIR / path).resolve())
    df = df.copy()
    df["file_path"] = df["file_path"].apply(_abs)
    return df


def report(original: pd.DataFrame, curated: pd.DataFrame, stats: dict):
    """Print before/after summary."""
    banner("CURATION SUMMARY")
    print(f"  Start:   {stats['start']:4d} rows  ({original['size_bytes'].sum()/1e9:.2f} GB)")
    for key, value in stats.items():
        if key.startswith("dropped_") and value > 0:
            label = key.replace("dropped_", "").replace("_", " ")
            print(f"  Dropped: {value:4d}  ({label})")
    print(f"  Kept:    {stats['kept']:4d} rows  ({curated['size_bytes'].sum()/1e9:.2f} GB)")

    banner("KEPT DOCS BY COMPANY")
    by_co = curated.groupby("company").agg(
        docs=("url", "count"),
        total_mb=("size_bytes", lambda x: round(x.sum() / 1e6, 1)),
        years=("fiscal_year", lambda x: f"{int(x.min()) if x.notna().any() else '-'}–{int(x.max()) if x.notna().any() else '-'}"),
    )
    print(by_co.to_string())

    banner("KEPT DOCS BY TYPE")
    by_type = curated["doc_type"].value_counts()
    for dt, n in by_type.items():
        bar = "█" * int(40 * n / len(curated))
        print(f"  {dt:25s} {n:4d}  {bar}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help=f"Source manifest (default: <script>/data/manifest.csv)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help=f"Curated output (default: <script>/data/curated_manifest.csv)")
    ap.add_argument("--include-secondary", action="store_true",
                    help="Also keep priority-2 (secondary) docs. Default: priority 1 only.")
    ap.add_argument("--include-xlsx", action="store_true",
                    help="Also keep xlsx ESG databooks. Default: PDFs only.")
    ap.add_argument("--min-year", type=int, default=2023,
                    help="Drop docs with fiscal_year < this (default: 2023)")
    ap.add_argument("--min-size-kb", type=int, default=500,
                    help="Drop files smaller than this in KB (default: 500)")
    ap.add_argument("--max-size-mb", type=int, default=80,
                    help="Drop files larger than this in MB (default: 80)")

    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        sys.exit(f"❌ Input manifest not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Filters: priority {'1+2' if args.include_secondary else '1'}, "
          f"year >= {args.min_year}, "
          f"size {args.min_size_kb}KB–{args.max_size_mb}MB, "
          f"format {'pdf+xlsx' if args.include_xlsx else 'pdf only'}")

    curated, stats = apply_filters(df.copy(), args)

    if len(curated) == 0:
        sys.exit("❌ Curation left 0 rows — relax the filters.")

    curated = absolutize_paths(curated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(output_path, index=False)

    report(df, curated, stats)
    print(f"\n✅ Wrote {len(curated)} curated rows to {output_path}")


if __name__ == "__main__":
    main()
