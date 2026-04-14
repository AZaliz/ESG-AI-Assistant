"""Command line interface for the ESG acquisition MVP."""

from __future__ import annotations

import argparse
import logging
import sys

from app.models import CompanySeed
from app.pipeline import AcquisitionPipeline
from app.smoke_test import run_smoke_test
from app.utils import ensure_directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESG document acquisition MVP")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover candidate reports for a company")
    discover.add_argument("--company", required=True, help="Company name")
    discover.add_argument("--ticker", help="Ticker")
    discover.add_argument("--country", help="Country")
    discover.add_argument("--issuer-domain", help="Known issuer domain")

    fetch = subparsers.add_parser("fetch", help="Discover and fetch the best report for a company")
    fetch.add_argument("--company", required=True, help="Company name")
    fetch.add_argument("--ticker", help="Ticker")
    fetch.add_argument("--country", help="Country")
    fetch.add_argument("--issuer-domain", help="Known issuer domain")

    smoke = subparsers.add_parser("smoke-test", help="Run the live smoke test")
    smoke.add_argument("--top-n", type=int, default=10, help="Number of European companies to test")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def seed_from_args(args: argparse.Namespace) -> CompanySeed:
    return CompanySeed(
        name=args.company,
        ticker=args.ticker,
        country=args.country,
        issuer_domain=args.issuer_domain,
        ranking_source="manual",
        ranking_url="manual",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    ensure_directories()

    if args.command == "smoke-test":
        rows, json_path, csv_path = run_smoke_test(top_n=args.top_n)
        print(f"Smoke test saved to {json_path} and {csv_path}")
        for row in rows:
            print(
                f"{row.company:20} | {row.chosen_report_title or 'NONE':40.40} | "
                f"{row.report_year or '-':4} | {row.confidence or 0:5.1f} | {row.parse_status or 'failed'}"
            )
        return 0

    pipeline = AcquisitionPipeline()
    seed = seed_from_args(args)
    if args.command == "discover":
        profile, candidates = pipeline.discover(seed)
        print(f"Resolved domains: {', '.join(profile.issuer_domains) or 'none'}")
        for candidate in candidates[:15]:
            print(f"{candidate.score:5.1f} | {candidate.document_type:22} | {candidate.url}")
        return 0

    if args.command == "fetch":
        _profile, record, notes = pipeline.fetch_best(seed)
        if not record:
            print("No document fetched.")
            for note in notes:
                print(f"- {note}")
            return 1
        print(record.model_dump_json(indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
