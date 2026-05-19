"""Quality and coverage check for the ESG scraper manifest.

Run after every scraper execution. Reports:
  1. Missing companies (vs the 10 expected targets)
  2. Per-company document inventory
  3. Critical coverage gaps (missing primary doc for current fiscal year)
  4. File integrity (size sanity, hash uniqueness, on-disk presence)
  5. Duplicate detection (same hash = same file with different metadata)
  6. Doc-type/title classification quality
  7. Suspicious entries needing manual review

Usage:
    python quality_check.py --manifest data/manifest.csv
    python quality_check.py --manifest data/manifest.csv --verify-files
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
import hashlib
import sys
from pathlib import Path

import pandas as pd


# Configuration ────────────────────────────────────────────────────────────

# Iberdrola dropped: site blocks non-residential IPs (Akamai WAF). Enel
# substituted as the European utility. See README "Note on coverage".
EXPECTED_COMPANIES = {
    "Airbus", "BNP Paribas", "Danone", "Enel", "Engie",
    "L'Oréal", "Schneider Electric", "Siemens", "TotalEnergies",
    "Volkswagen",
}

# A "primary annual disclosure" — every company should have at least one of these
# for the most recent fiscal year, otherwise the corpus has a hole for that company.
PRIMARY_DOC_TYPES = {
    "urd", "annual_report", "sustainability_report",
    "integrated_report", "progress_report",
}

# Minimum file size (bytes) per file_format below which we suspect a download
# failure / error page. ESG databooks (.xlsx) are legitimately small spreadsheets,
# so applying the PDF threshold to them produces ~11 false alarms.
MIN_SIZE_BY_FORMAT = {
    "pdf":  200_000,   # 200 KB — real ESG reports are always larger
    "xlsx":  20_000,   # 20 KB — Engie ESG databooks are 47–69 KB
    "xls":   20_000,
    "csv":    5_000,
    "html":  20_000,
}
DEFAULT_MIN_SIZE = 100_000

# Back-compat — the summary block still references a single threshold
MIN_REASONABLE_SIZE = MIN_SIZE_BY_FORMAT["pdf"]


def _min_size_for(file_format: str) -> int:
    return MIN_SIZE_BY_FORMAT.get((file_format or "pdf").lower(), DEFAULT_MIN_SIZE)

# Current fiscal year for coverage check (most recent CSRD-mandated year)
CURRENT_FY = 2024


# ──────────────────────────────────────────────────────────────────────────
# Check functions
# ──────────────────────────────────────────────────────────────────────────

def banner(s: str):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


def check_company_coverage(df: pd.DataFrame) -> list:
    """Are all 10 expected companies represented?"""
    banner("1. COMPANY COVERAGE")
    found = set(df["company"].unique())
    missing = EXPECTED_COMPANIES - found
    extra = found - EXPECTED_COMPANIES

    print(f"Expected:  {len(EXPECTED_COMPANIES)} companies")
    print(f"Found:     {len(found)} companies")
    if missing:
        print(f"❌ MISSING: {sorted(missing)}")
    else:
        print("✅ All expected companies present")
    if extra:
        print(f"⚠️  Unexpected (typo?): {sorted(extra)}")
    return sorted(missing)


def check_per_company_inventory(df: pd.DataFrame):
    """Documents per company, with year range and total volume."""
    banner("2. PER-COMPANY INVENTORY")
    per_co = df.groupby("company").agg(
        docs=("url", "count"),
        primary=("priority", lambda x: (x == 1).sum()),
        secondary=("priority", lambda x: (x == 2).sum()),
        archive=("priority", lambda x: (x == 3).sum()),
        min_year=("fiscal_year", "min"),
        max_year=("fiscal_year", "max"),
        total_mb=("size_bytes", lambda x: round(x.sum() / 1e6, 1)),
    )
    print(per_co.to_string())

    print("\nFlags:")
    for co, row in per_co.iterrows():
        if row["docs"] < 3:
            print(f"  ⚠️  {co}: only {row['docs']} docs (suspiciously few)")
        if row["primary"] == 0:
            print(f"  ❌ {co}: NO primary-priority documents")


def check_critical_coverage(df: pd.DataFrame, fy: int = CURRENT_FY) -> dict:
    """For each company, is there at least one primary doc for the current FY?"""
    banner(f"3. CRITICAL COVERAGE FOR FY{fy}")
    gaps = {}
    print(f"Every company must have ≥1 primary doc ({', '.join(sorted(PRIMARY_DOC_TYPES))}) for FY{fy}.\n")

    for co in sorted(df["company"].unique()):
        sub = df[(df["company"] == co) &
                 (df["fiscal_year"] == fy) &
                 (df["doc_type"].isin(PRIMARY_DOC_TYPES))]
        if len(sub) == 0:
            print(f"  ❌ {co}: NO primary disclosure for FY{fy}")
            gaps[co] = []
        else:
            types = sorted(sub["doc_type"].unique())
            print(f"  ✅ {co}: {types}")
    return gaps


def check_duplicates(df: pd.DataFrame):
    """Same content (same hash) appearing under multiple rows."""
    banner("4. DUPLICATE CONTENT (same hash, multiple rows)")
    dups = df[df.duplicated(subset="file_hash", keep=False)].sort_values(["file_hash", "company"])
    n_groups = dups["file_hash"].nunique()
    print(f"Found {len(dups)} rows in {n_groups} duplicate groups.\n")

    if n_groups == 0:
        print("✅ No duplicates")
        return

    for h, g in dups.groupby("file_hash"):
        size_mb = g["size_bytes"].iloc[0] / 1e6
        print(f"\n  Hash {h[:12]}… ({len(g)} rows, {size_mb:.1f} MB):")
        for _, r in g.iterrows():
            print(f"    [{r['company']:14s}] {r['doc_type']:22s} "
                  f"fy={int(r['fiscal_year']) if pd.notna(r['fiscal_year']) else '?':>4} "
                  f"title={r['title'][:50]!r}")
        print("    → Consider deduplicating in manifest (same file, redundant metadata).")


def check_file_sizes(df: pd.DataFrame):
    """Files that look too small to be real reports."""
    banner("5. FILE SIZE SANITY")
    if df["size_bytes"].isna().all():
        print("⚠️  No size data in manifest")
        return

    print(f"Size distribution (MB):")
    print(f"  min:    {df['size_bytes'].min() / 1e6:.2f}")
    print(f"  median: {df['size_bytes'].median() / 1e6:.2f}")
    print(f"  mean:   {df['size_bytes'].mean() / 1e6:.2f}")
    print(f"  max:    {df['size_bytes'].max() / 1e6:.2f}")

    df["_min_size"] = df["file_format"].apply(_min_size_for)
    too_small = df[df["size_bytes"] < df["_min_size"]]
    if len(too_small):
        thresholds = ", ".join(f"{f}≥{s/1e3:.0f}KB" for f, s in sorted(MIN_SIZE_BY_FORMAT.items()))
        print(f"\n❌ {len(too_small)} file(s) below per-format threshold ({thresholds}):")
        for _, r in too_small.iterrows():
            print(f"  [{r['company']:14s}] {r['title'][:40]:42s} "
                  f"{r['size_bytes'] / 1e3:>5.0f} KB  {r['doc_type']:22s} ({r['file_format']})")
            print(f"      url: {r['url'][:90]}")
    else:
        print("\n✅ All files above per-format size thresholds")


def check_missing_metadata(df: pd.DataFrame):
    """Rows where critical fields (year, type) are missing."""
    banner("6. METADATA QUALITY")

    no_year = df[df["fiscal_year"].isna()]
    print(f"Rows with no fiscal_year: {len(no_year)}")
    if len(no_year) and len(no_year) <= 15:
        for _, r in no_year.iterrows():
            print(f"  [{r['company']:14s}] {r['title'][:50]:52s} {r['doc_type']}")

    # Generic / acronym-only titles indicate the scraper extracted link text poorly
    suspect = df[df["title"].str.match(r"^([A-Z]{2,5}|PDF|XLSX)$", na=False)]
    print(f"\nRows with generic/acronym-only titles: {len(suspect)}")
    if len(suspect):
        for _, r in suspect.iterrows():
            slug = r["url"].rsplit("/", 1)[-1][:60]
            print(f"  [{r['company']:14s}] title={r['title']!r:8s} doc_type={r['doc_type']:22s} url={slug}")
        print("  → These were classified by URL pattern; titles can be improved in extractor.")


def check_doctype_balance(df: pd.DataFrame):
    """How is the corpus split across doc_types?"""
    banner("7. DOC_TYPE BALANCE")
    counts = df["doc_type"].value_counts()
    total = len(df)
    for dt, n in counts.items():
        bar = "█" * int(40 * n / total)
        print(f"  {dt:25s} {n:4d}  {bar}")


def verify_files_on_disk(df: pd.DataFrame, manifest_dir: Path):
    """Optionally verify each file exists on disk and matches its recorded hash."""
    banner("8. FILE INTEGRITY (--verify-files)")
    n_ok, n_missing, n_corrupt = 0, 0, 0
    for _, r in df.iterrows():
        p = manifest_dir / r["file_path"]
        if not p.exists():
            # also try as absolute path
            p = Path(r["file_path"])
            if not p.exists():
                print(f"  ❌ MISSING: {r['file_path']}")
                n_missing += 1
                continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != r["file_hash"]:
            print(f"  ❌ CORRUPT: {r['file_path']}")
            print(f"      recorded:  {r['file_hash'][:16]}…")
            print(f"      on disk:   {actual[:16]}…")
            n_corrupt += 1
        else:
            n_ok += 1
    print(f"\n  ✅ {n_ok} files match | ❌ {n_missing} missing | ❌ {n_corrupt} corrupt")


# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--verify-files", action="store_true",
                    help="Also recompute hashes of files on disk (slow)")
    ap.add_argument("--fiscal-year", type=int, default=CURRENT_FY)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} rows from {manifest_path}")

    check_company_coverage(df)
    check_per_company_inventory(df)
    check_critical_coverage(df, fy=args.fiscal_year)
    check_duplicates(df)
    check_file_sizes(df)
    check_missing_metadata(df)
    check_doctype_balance(df)
    if args.verify_files:
        verify_files_on_disk(df, manifest_path.parent.parent)

    banner("SUMMARY")
    found = set(df["company"].unique())
    missing = EXPECTED_COMPANIES - found
    fy_gaps = [
        co for co in sorted(df["company"].unique())
        if len(df[(df["company"] == co)
                  & (df["fiscal_year"] == args.fiscal_year)
                  & (df["doc_type"].isin(PRIMARY_DOC_TYPES))]) == 0
    ]
    small = (df["size_bytes"] < df["file_format"].apply(_min_size_for)).sum()

    print(f"  Companies missing entirely:           {len(missing)}")
    print(f"  Companies w/o FY{args.fiscal_year} primary disclosure: {len(fy_gaps)}")
    print(f"  Files below per-format threshold:     {small}")
    print(f"  Total documents:                      {len(df)}")
    print(f"  Total volume:                         {df['size_bytes'].sum() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()