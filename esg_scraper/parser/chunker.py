"""Chunker: build RAG-ready chunks from parsed pages + normalized table rows.

Each chunk includes:
  - text: what gets embedded
  - metadata: company, doc_type, fiscal_year, publication_year, page, source_file,
              chunk_kind ('narrative' | 'table_fact'), source_authority

Two chunk kinds:
  - 'narrative': contiguous page text, sized to target_tokens with overlap
  - 'table_fact': one normalized sentence per fact (no chunking — already atomic)

Table_fact chunks are typically much shorter than narrative chunks but their
embeddings are much sharper because each contains exactly one fact with
explicit subject/unit/year. The audit's "value present in table" PARTIALs
are mostly resolved by indexing these.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict, field
from typing import Any

from parser.taxonomy import classify_esrs_topics

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One unit of indexable content."""
    chunk_id: str
    text: str
    company: str
    doc_type: str
    fiscal_year: int | None
    publication_year: int | None
    page: int
    source_file: str
    chunk_kind: str  # 'narrative' or 'table_fact'
    source_authority: int = 1  # 1=company primary, 2=company supplement, 3=regulation
    # Table-fact only
    table_id: str | None = None
    raw_label: str | None = None
    raw_value: str | None = None
    raw_unit: str | None = None
    # Tagging (parser.taxonomy). speaker_role is None for every non-
    # earnings_call chunk; esrs_topic is [] when no topic matches.
    speaker_role: str | None = None
    esrs_topic: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _count_tokens_approx(text: str) -> int:
    """Approximate token count without depending on tiktoken (≈1 token / 0.75 words)."""
    return int(len(text.split()) / 0.75)


def _chunk_narrative(
    text: str,
    page: int,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split narrative page text into overlapping chunks.

    Word-based sliding window. ~400 tokens ≈ 300 words (rough).
    """
    if not text or not text.strip():
        return []

    words = text.split()
    target_words = int(target_tokens / 0.75)
    overlap_words = int(overlap_tokens / 0.75)

    if len(words) <= target_words:
        return [" ".join(words)]

    chunks = []
    i = 0
    while i < len(words):
        window = words[i : i + target_words]
        chunks.append(" ".join(window))
        i += target_words - overlap_words
    return chunks


def build_chunks_for_document(
    pages: list,                  # list of ExtractedPage
    normalized_table_rows: list,  # list of NormalizedRow
    doc_meta: dict,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Produce a complete chunk list for one document.

    doc_meta must contain: company, doc_type, fiscal_year, publication_year,
    source_file, source_authority.
    """
    required = ["company", "doc_type", "fiscal_year", "publication_year",
                "source_file", "source_authority"]
    missing = [k for k in required if k not in doc_meta]
    if missing:
        raise ValueError(f"doc_meta missing keys: {missing}")

    chunks = []
    doc_id_base = re.sub(r"[^a-z0-9]+", "_", doc_meta["source_file"].lower())[:40]
    counter = 0

    # 1) Narrative chunks — one or more per page
    for page in pages:
        if not page.text or not page.text.strip():
            continue
        for chunk_text in _chunk_narrative(page.text, page.page, target_tokens, overlap_tokens):
            counter += 1
            chunks.append(Chunk(
                chunk_id=f"{doc_id_base}_n{counter:04d}",
                text=chunk_text,
                company=doc_meta["company"],
                doc_type=doc_meta["doc_type"],
                fiscal_year=doc_meta["fiscal_year"],
                publication_year=doc_meta["publication_year"],
                page=page.page,
                source_file=doc_meta["source_file"],
                chunk_kind="narrative",
                source_authority=doc_meta["source_authority"],
                esrs_topic=classify_esrs_topics(chunk_text, "narrative"),
            ))

    # 2) Table fact chunks — one per normalized row (no further chunking)
    for nr in normalized_table_rows:
        counter += 1
        chunks.append(Chunk(
            chunk_id=f"{doc_id_base}_t{counter:04d}",
            text=nr.sentence,
            company=doc_meta["company"],
            doc_type=doc_meta["doc_type"],
            fiscal_year=doc_meta["fiscal_year"],
            publication_year=doc_meta["publication_year"],
            page=nr.page,
            source_file=doc_meta["source_file"],
            chunk_kind="table_fact",
            source_authority=doc_meta["source_authority"],
            table_id=nr.table_id,
            raw_label=nr.raw_label,
            raw_value=nr.raw_value,
            raw_unit=nr.raw_unit,
            esrs_topic=classify_esrs_topics(nr.sentence, "table_fact", nr.raw_label),
        ))

    return chunks


# ──────────────────────────────────────────────────────────────────────────
# Stats for diagnostics
# ──────────────────────────────────────────────────────────────────────────

def chunk_stats(chunks: list[Chunk]) -> dict:
    """Compute basic stats for a chunk list."""
    if not chunks:
        return {"total": 0}

    narrative = [c for c in chunks if c.chunk_kind == "narrative"]
    tables = [c for c in chunks if c.chunk_kind == "table_fact"]
    avg_n_tokens = sum(_count_tokens_approx(c.text) for c in narrative) / max(len(narrative), 1)
    avg_t_tokens = sum(_count_tokens_approx(c.text) for c in tables) / max(len(tables), 1)

    return {
        "total": len(chunks),
        "narrative": len(narrative),
        "table_fact": len(tables),
        "avg_narrative_tokens": round(avg_n_tokens, 1),
        "avg_table_fact_tokens": round(avg_t_tokens, 1),
        "table_share_pct": round(100 * len(tables) / len(chunks), 1),
    }
