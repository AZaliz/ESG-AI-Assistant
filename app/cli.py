"""Command line interface for the ESG acquisition MVP."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from app.models import CompanySeed
from app.pipeline import AcquisitionPipeline
from app.rag import DEFAULT_BASE_URL, DEFAULT_INDEX_DIR, run_rag_ask, run_rag_build
from app.rag_eval import DEFAULT_EVAL_DATASET, DEFAULT_EVAL_OUTPUT, run_rag_eval
from app.smoke_test import run_smoke_test
from app.utils import ensure_directories
from app.web import run_web_app


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

    rag_build = subparsers.add_parser("rag-build", help="Build a local ESG RAG index from PDF files")
    rag_build.add_argument(
        "--pdf",
        action="append",
        help="Path to a PDF to index. Repeat the flag to include multiple files.",
    )
    rag_build.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory where chunk metadata and vectors should be stored.",
    )
    rag_build.add_argument(
        "--chunk-target-tokens",
        type=int,
        default=420,
        help="Preferred chunk size in approximate tokens.",
    )
    rag_build.add_argument(
        "--chunk-min-tokens",
        type=int,
        default=300,
        help="Minimum chunk size in approximate tokens.",
    )
    rag_build.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=500,
        help="Maximum chunk size in approximate tokens.",
    )
    rag_build.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many chunks to send per embeddings request.",
    )
    rag_build.add_argument(
        "--embedding-model",
        help="Albert embedding model id. Defaults to a detected BGE-M3-compatible model.",
    )
    rag_build.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )
    rag_build.add_argument(
        "--dry-run",
        action="store_true",
        help="Only extract and chunk the PDFs without calling the Albert API.",
    )

    rag_ask = subparsers.add_parser("rag-ask", help="Ask grounded questions against a local ESG RAG index")
    rag_ask.add_argument(
        "question",
        nargs="+",
        help="Question to ask against the indexed ESG report chunks.",
    )
    rag_ask.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the built RAG index.",
    )
    rag_ask.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many chunks to retrieve before answering.",
    )
    rag_ask.add_argument(
        "--embedding-model",
        help="Albert embedding model id for the question embedding.",
    )
    rag_ask.add_argument(
        "--text-model",
        help="Albert text-generation model id for answer generation.",
    )
    rag_ask.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )
    rag_ask.add_argument(
        "--search-only",
        action="store_true",
        help="Only retrieve chunks and skip the final LLM answer step.",
    )

    rag_web = subparsers.add_parser("rag-web", help="Launch the local browser UI for ESG RAG")
    rag_web.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the local web app to.",
    )
    rag_web.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port for the local web app.",
    )
    rag_web.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the RAG index used by the web app.",
    )

    rag_eval = subparsers.add_parser("rag-eval", help="Evaluate retrieval quality against the ESG QA dataset")
    rag_eval.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the built RAG index.",
    )
    rag_eval.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="CSV dataset containing ESG evaluation questions and expected contexts.",
    )
    rag_eval.add_argument(
        "--company",
        help="Company to evaluate. If omitted, the command tries to infer it from the indexed PDF paths.",
    )
    rag_eval.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many chunks to retrieve for each evaluation question.",
    )
    rag_eval.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT,
        help="Where to save the evaluation JSON output.",
    )
    rag_eval.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )

    streamlit_ui = subparsers.add_parser("streamlit-ui", help="Launch the Streamlit prompt console")
    streamlit_ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Streamlit app.",
    )
    streamlit_ui.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit app.",
    )
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

    if args.command == "rag-build":
        return run_rag_build(args)

    if args.command == "rag-ask":
        return run_rag_ask(args)

    if args.command == "rag-web":
        run_web_app(host=args.host, port=args.port, index_dir=args.index_dir)
        return 0

    if args.command == "rag-eval":
        return run_rag_eval(args)

    if args.command == "streamlit-ui":
        streamlit_path = Path(__file__).with_name("streamlit_ui.py")
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(streamlit_path),
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ]
        return subprocess.call(command)

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
