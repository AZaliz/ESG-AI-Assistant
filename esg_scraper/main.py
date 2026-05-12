"""ESG Scraper — main entry point.

Usage:
    # Dry run on all companies (discover only, no downloads)
    python main.py --all --dry-run

    # Download everything for one company
    python main.py --company totalenergies

    # Multiple companies, parallel
    python main.py --company engie --company bnp_paribas --workers 3

    # All 10 companies, full download
    python main.py --all
"""
import argparse
import csv
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scraper.config import SCRAPERS
from scraper.models import Document, DOC_TYPES, ESG_REPORT_DOC_TYPES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Defaults resolve relative to this file, NOT the caller's cwd. Without this,
# running `python /full/path/to/main.py --all` from elsewhere drops data/ next
# to the caller, which split the corpus across two trees the first time it bit us.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "pdfs"
DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "manifest.csv"


def run_one(scraper_class, output_dir: Path, dry_run: bool, allowed_doc_types: set):
    scraper = scraper_class(output_dir)
    return scraper.run(dry_run=dry_run, allowed_doc_types=allowed_doc_types)


def write_manifest(docs, path: Path):
    if not docs:
        logger.warning("No documents to write to manifest")
        return
    rows = [d.to_manifest_row() for d in docs]
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Wrote manifest with {len(rows)} entries to {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Scrape ESG documents from corporate sustainability sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available companies: {', '.join(sorted(SCRAPERS.keys()))}",
    )
    ap.add_argument("--company", action="append",
                    help="Scrape one company (repeatable). E.g. --company totalenergies")
    ap.add_argument("--all", action="store_true",
                    help="Scrape every registered company")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover only, don't download")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Where to save PDFs (default: <script_dir>/data/pdfs)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="Where to save the manifest CSV (default: <script_dir>/data/manifest.csv)")
    ap.add_argument("--workers", type=int, default=3,
                    help="Parallel companies (default: 3)")
    ap.add_argument("--list", action="store_true",
                    help="List available companies and exit")
    ap.add_argument("--with-policies", action="store_true",
                    help="Include policy docs (codes of conduct, supplier codes, "
                         "tax strategy, etc.). Default: ESG reports only.")
    args = ap.parse_args()

    if args.list:
        print("Available companies:")
        for key, cls in sorted(SCRAPERS.items()):
            print(f"  {key:18s} → {cls.company}")
        return 0

    if args.all:
        targets = list(SCRAPERS.keys())
    elif args.company:
        targets = []
        for c in args.company:
            if c not in SCRAPERS:
                logger.error(f"Unknown company: {c}. Use --list to see options.")
                return 1
            targets.append(c)
    else:
        ap.error("Pass --all or --company X (use --list to see options)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_doc_types = DOC_TYPES if args.with_policies else ESG_REPORT_DOC_TYPES

    logger.info(f"Targets: {', '.join(targets)}")
    logger.info(f"Output:  {output_dir}")
    if args.dry_run:
        logger.info("Mode:    DRY RUN (no downloads)")
    if args.with_policies:
        logger.info("Scope:   ALL doc_types (policies included)")
    else:
        logger.info(f"Scope:   ESG reports only ({len(allowed_doc_types)} doc_types; policies excluded)")

    all_docs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, SCRAPERS[name], output_dir, args.dry_run, allowed_doc_types): name
            for name in targets
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                docs = fut.result()
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"[{name}] crashed: {e}", exc_info=True)

    if not args.dry_run and all_docs:
        write_manifest(all_docs, Path(args.manifest))

    print(f"\n{'='*60}")
    print(f"Done: {len(all_docs)} documents across {len(targets)} companies")
    if args.dry_run:
        print("(dry run — nothing was downloaded)")
    else:
        print(f"Files in: {output_dir}")
        print(f"Manifest: {args.manifest}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
