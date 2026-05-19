"""Build the indexable chunk set from the scraper manifest.

Reads data/manifest.csv, picks the documents that pass the indexing filter
(Tier 1 + Tier 2 from the document-weighting doc), parses each one with text+
table extraction, normalizes the tables, chunks everything, and writes:

    data/chunks/chunks.jsonl              # one chunk per line, ready for embedding
    data/chunks/chunks_index.csv          # summary index (chunk_id, page, company, ...)
    data/chunks/build_report.md           # per-document stats: tables found, etc.

Usage:
    python -m parser.build_index
    python -m parser.build_index --manifest data/manifest.csv --output data/chunks
    python -m parser.build_index --company totalenergies        # one company only
    python -m parser.build_index --dry-run                       # report what would be parsed
"""
from __future__ import annotations

# --- UTF-8 console bootstrap: Windows wraps stdout in cp1252, which can't
# encode →, —, É (L'Oréal), etc. Force UTF-8 so output never crashes. ---
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .parser import parse_pdf_to_list
from .table_normalizer import normalize_all_tables, summarize_normalization
from .chunker import build_chunks_for_document, chunk_stats
from .corpus_io import splice_chunks, existing_source_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# URLs whose docs must be parsed text-only (registry lives in eurlex.py).
# Soft import: build_index must still run if the scraper package is absent.
try:
    from scraper.extractors.eurlex import NARRATIVE_ONLY_URLS as _EURLEX_NARRATIVE
except Exception:  # pragma: no cover - scraper pkg optional for parsing
    _EURLEX_NARRATIVE = frozenset()

# Non-regulation docs that must also be parsed text-only. The Airbus ESG
# Datasheet is an infographic-style PDF: pdfplumber's table detector mis-grids
# it (words shredded across phantom cells, no intact KPI column), but its text
# layer is clean prose. Same pathology/fix as the CSDDD. Keyed by the stable
# mediaassets URL so it survives re-chunks.
_EXTRA_NARRATIVE_URLS = frozenset({
    "https://mediaassets.airbus.com/pm_38_711_711991-us758j55ai.pdf",
})
NARRATIVE_ONLY_URLS = _EURLEX_NARRATIVE | _EXTRA_NARRATIVE_URLS


# Tier 1 + Tier 2 selection rules
TIER_1_TYPES = {"urd", "annual_report", "sustainability_report",
                "integrated_report", "progress_report"}
TIER_2_TYPES = {"esg_databook", "climate_report", "vigilance_plan"}
TIER_3_TYPES = {"regulation"}   # definitional anchors (ESRS, GHG Protocol, etc.)
INDEXABLE_DOC_TYPES = TIER_1_TYPES | TIER_2_TYPES | TIER_3_TYPES
TIER_1_YEARS = {2022, 2023, 2024}


def filter_indexable(manifest: pd.DataFrame) -> pd.DataFrame:
    """Apply the indexing filter from the document-weighting doc:
       - priority == 1
       - fiscal_year in {2022, 2023, 2024}
       - doc_type in indexable set
       - file exists on disk
    """
    df = manifest.copy()

    # Coerce types
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")

    mask = (
        (df["priority"] == 1)
        & (df["doc_type"].isin(INDEXABLE_DOC_TYPES))
        # Regulations have no year restriction (ESRS 2023, GHG Protocol 2004,
        # IEA NZE 2021 are all evergreen). Company docs use TIER_1_YEARS.
        & (
            df["doc_type"].isin(TIER_3_TYPES)
            | df["fiscal_year"].isin(TIER_1_YEARS)
        )
        & (df["file_path"].notna())
        & (df["file_path"] != "")
    )
    selected = df[mask].copy()
    logger.info(f"  filter: {len(manifest)} → {len(selected)} indexable documents")
    return selected


def source_authority_for(doc_type: str) -> int:
    """Assign source_authority based on doc_type.
       1 = primary company disclosure
       2 = company supplement
       3 = regulation/standard (handled elsewhere, but defined for completeness)
    """
    if doc_type in TIER_1_TYPES:
        return 1
    if doc_type in TIER_2_TYPES:
        return 2
    return 3  # regulation_text, framework_standard, etc.


def process_one_document(row: pd.Series, max_pages: int | None = None) -> tuple[list, dict]:
    """Parse one document and produce its chunks.

    Returns (chunks, report_dict).
    """
    file_path = Path(row["file_path"])
    if not file_path.exists():
        return [], {"file": row["file_path"], "error": "file not found", "chunks": 0}

    # Only PDFs for now (xlsx databooks need a separate xlsx_parser — TODO)
    if not file_path.suffix.lower() == ".pdf":
        return [], {
            "file": file_path.name, "doc_type": row["doc_type"],
            "skipped": "non-pdf format (xlsx normalization in separate module)",
            "chunks": 0,
        }

    narrative_only = str(row.get("url", "")) in NARRATIVE_ONLY_URLS
    if narrative_only:
        logger.info(f"  narrative_only=True (text-only parse) for {file_path.name}")

    t0 = time.time()
    try:
        pages = parse_pdf_to_list(
            file_path, max_pages=max_pages, narrative_only=narrative_only
        )
    except Exception as e:
        logger.warning(f"  parse failed: {file_path.name}: {e}")
        return [], {"file": file_path.name, "error": str(e), "chunks": 0}

    n_tables = sum(len(p.tables) for p in pages)
    n_pages = len(pages)

    doc_meta = {
        "company": row["company"],
        "doc_type": row["doc_type"],
        "fiscal_year": int(row["fiscal_year"]) if pd.notna(row["fiscal_year"]) else None,
        "publication_year": int(row["publication_year"]) if pd.notna(row["publication_year"]) else None,
        "source_file": file_path.name,
        "source_authority": source_authority_for(row["doc_type"]),
    }

    normalized = normalize_all_tables(pages, doc_meta)
    chunks = build_chunks_for_document(pages, normalized, doc_meta)

    elapsed = time.time() - t0
    report = {
        "file": file_path.name,
        "company": row["company"],
        "doc_type": row["doc_type"],
        "fiscal_year": doc_meta["fiscal_year"],
        "pages": n_pages,
        "tables_found": n_tables,
        "normalized_facts": len(normalized),
        "narrative_chunks": sum(1 for c in chunks if c.chunk_kind == "narrative"),
        "table_fact_chunks": sum(1 for c in chunks if c.chunk_kind == "table_fact"),
        "total_chunks": len(chunks),
        "elapsed_sec": round(elapsed, 1),
    }
    return chunks, report


def write_outputs(all_chunks: list, reports: list, out_dir: Path):
    """Write chunks.jsonl, chunks_index.csv, build_report.md (FULL rebuild).

    This is a full overwrite: a `--company`/filtered run here would replace
    the ENTIRE corpus with just the selected docs. For targeted/additive
    updates use `--incremental` (main()), which appends via
    corpus_io.splice_chunks instead of calling this. This function is the
    intentional full-rebuild path only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # chunks.jsonl — one chunk per line, ready for embedding
    jsonl_path = out_dir / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"  wrote {len(all_chunks)} chunks to {jsonl_path}")

    # chunks_index.csv — flat view for quick filtering in pandas
    if all_chunks:
        index_path = out_dir / "chunks_index.csv"
        rows = []
        for c in all_chunks:
            d = c.to_dict()
            d["text_preview"] = d["text"][:120].replace("\n", " ")
            d.pop("text")
            rows.append(d)
        pd.DataFrame(rows).to_csv(index_path, index=False)
        logger.info(f"  wrote chunk index to {index_path}")

    # build_report.md — human-readable per-document summary
    report_path = out_dir / "build_report.md"
    report_df = pd.DataFrame(reports)
    md_lines = ["# Chunk Build Report\n"]
    md_lines.append(f"**Processed:** {len(reports)} documents → {len(all_chunks)} chunks\n")
    if len(report_df):
        narrative_total = report_df["narrative_chunks"].sum() if "narrative_chunks" in report_df else 0
        table_total = report_df["table_fact_chunks"].sum() if "table_fact_chunks" in report_df else 0
        tables_found = report_df["tables_found"].sum() if "tables_found" in report_df else 0
        md_lines.append(f"\n**Narrative chunks:** {narrative_total}")
        md_lines.append(f"\n**Table-fact chunks:** {table_total}")
        md_lines.append(f"\n**Tables detected (raw):** {tables_found}")
        md_lines.append("\n\n## Per-document detail\n")
        md_lines.append(report_df.to_markdown(index=False))
    Path(report_path).write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"  wrote build report to {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--output", default="data/chunks")
    ap.add_argument("--company", help="Process one company only (e.g. totalenergies)")
    ap.add_argument("--max-pages", type=int, help="Cap pages per document (for testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be processed; do not parse")
    ap.add_argument("--incremental", action="store_true",
                    help="Only process manifest docs whose source_file is not "
                         "already in the corpus; APPEND via splice instead of "
                         "overwriting chunks.jsonl/_index.csv. Safe for adding "
                         "newly-scraped docs without a full rebuild.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return 1

    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest: {len(df)} rows")

    selected = filter_indexable(df)
    if args.company:
        before = len(selected)
        selected = selected[selected["company"].str.lower().str.contains(args.company.lower())]
        logger.info(f"  company filter '{args.company}': {before} → {len(selected)}")

    if args.incremental:
        have = existing_source_files(Path(args.output))
        before = len(selected)
        selected = selected[
            ~selected["file_path"].apply(lambda p: Path(str(p)).name).isin(have)
        ]
        logger.info(f"  incremental: {before} → {len(selected)} not-yet-chunked "
                    f"({len(have)} source_files already in corpus)")
        if selected.empty:
            logger.info("Nothing new to chunk — corpus already up to date.")
            return 0

    if selected.empty:
        logger.warning("No indexable documents after filtering")
        return 1

    if args.dry_run:
        print("\nWould process:")
        for _, row in selected.iterrows():
            fy = f"{int(row['fiscal_year']):.0f}" if pd.notna(row['fiscal_year']) else "----"
            print(f"  {row['company']:22s} {fy} {row['doc_type']:22s} {Path(row['file_path']).name}")
        print(f"\nTotal: {len(selected)} documents")
        return 0

    all_chunks = []
    reports = []
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        fy_label = f"FY{int(row['fiscal_year'])}" if pd.notna(row['fiscal_year']) else "n/a"
        logger.info(f"[{i}/{len(selected)}] {row['company']} | {row['doc_type']} | {fy_label}")
        chunks, report = process_one_document(row, max_pages=args.max_pages)
        all_chunks.extend(chunks)
        reports.append(report)
        if chunks:
            stats = chunk_stats(chunks)
            logger.info(f"  → {stats['narrative']} narrative + {stats['table_fact']} table_fact "
                        f"= {stats['total']} chunks ({stats['table_share_pct']}% table-fact)")

    n_table = sum(1 for c in all_chunks if c.chunk_kind == "table_fact")
    n_narr = sum(1 for c in all_chunks if c.chunk_kind == "narrative")

    if args.incremental:
        if not all_chunks:
            # Selected docs produced nothing (e.g. xlsx databooks, which the
            # PDF parser skips). Not an error; corpus left untouched.
            print("No new chunks produced (selected docs yielded 0 chunks, "
                  "e.g. .xlsx). Corpus unchanged.")
            return 0
        # Append-only: never touches the other docs already in the corpus.
        # Splice only source_files that actually produced chunks.
        src = sorted({c.source_file for c in all_chunks})
        stats = splice_chunks(
            all_chunks, src, chunk_dir=Path(args.output),
            backup_suffix=".incremental.bak",
        )
        rpt = Path(args.output) / "build_report_incremental.md"
        rdf = pd.DataFrame(reports)
        lines = [f"# Incremental build — added {len(selected)} document(s)\n",
                 f"**New chunks:** {len(all_chunks)} "
                 f"(narrative {n_narr}, table_fact {n_table})\n",
                 f"**Corpus now:** {stats['csv_rows']} rows\n"]
        if len(rdf):
            lines.append("\n## Added documents\n")
            lines.append(rdf.to_markdown(index=False))
        rpt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"Incremental: +{len(all_chunks)} chunks from {len(selected)} new docs")
        print(f"  corpus now: {stats['csv_rows']} rows (jsonl==csv, no dup ids)")
        print(f"  report:     {rpt}")
        print(f"{'='*60}")
        return 0

    write_outputs(all_chunks, reports, Path(args.output))

    # Final summary
    print(f"\n{'='*60}")
    print(f"Done: {len(all_chunks)} chunks from {len(selected)} documents")
    print(f"  narrative:  {n_narr:5d}")
    print(f"  table_fact: {n_table:5d}  ({100*n_table/max(len(all_chunks),1):.1f}%)")
    print(f"  output:     {args.output}/")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
