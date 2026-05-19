"""Table-aware document parser for ESG corpus.

Solves the audit's #1 finding: ~75% of PARTIAL data points are figures locked
in tables that text-only extraction can't reconstruct. This module extracts
text and tables together, normalizes tables to natural-language sentences,
and produces RAG-ready chunks.

Usage:
    python -m parser.build_index --manifest data/manifest.csv

Modules:
    parser.py            — pdfplumber-based text + table extraction
    table_normalizer.py  — table rows → natural-language sentences
    chunker.py           — combine narrative + table_fact into typed chunks
    build_index.py       — orchestrator: manifest → chunks.jsonl
"""
__version__ = "0.1.0"
