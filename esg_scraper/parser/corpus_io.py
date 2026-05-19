"""Shared chunk-corpus append/splice.

Single source of truth for mutating data/chunks/{chunks.jsonl,
chunks_index.csv} WITHOUT a full rebuild. Used by:
  - build_index.py  --incremental   (add manifest docs not yet chunked)
  - rechunk_doc.py                   (re-chunk specific source_files)
  - ingest_earnings.py               (feed the .txt transcript corpus)

Guarantees: one-time *.bak backup per suffix, the named source_files are
removed before re-adding (idempotent), chunks_index.csv column order is
preserved, jsonl and csv stay row-aligned, and chunk_ids are asserted unique.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

CHUNK_DIR = Path("data/chunks")


def _index_row(chunk_dict: dict) -> dict:
    d = dict(chunk_dict)
    d["text_preview"] = d["text"][:120].replace("\n", " ")
    d.pop("text")
    return d


def existing_source_files(chunk_dir: Path = CHUNK_DIR) -> set[str]:
    """source_file values already present in the corpus index."""
    csv = chunk_dir / "chunks_index.csv"
    if not csv.exists():
        return set()
    return set(pd.read_csv(csv, usecols=["source_file"])["source_file"].unique())


def splice_chunks(
    new_chunks: list,
    source_files: list[str],
    chunk_dir: Path = CHUNK_DIR,
    backup_suffix: str = ".bak",
) -> dict:
    """Replace all rows for `source_files` with `new_chunks` (append if new).

    Returns stats dict. Raises AssertionError on duplicate chunk_ids.
    """
    chunk_dir = Path(chunk_dir)
    jsonl_p = chunk_dir / "chunks.jsonl"
    csv_p = chunk_dir / "chunks_index.csv"
    targets = set(source_files)

    for f in (jsonl_p, csv_p):
        bak = f.with_suffix(f.suffix + backup_suffix)
        if not bak.exists():
            shutil.copy(f, bak)

    # jsonl: drop old rows for these source_files, append new
    kept = [
        ln for ln in jsonl_p.read_text(encoding="utf-8").splitlines()
        if ln and json.loads(ln).get("source_file") not in targets
    ]
    kept.extend(json.dumps(c.to_dict(), ensure_ascii=False) for c in new_chunks)
    jsonl_p.write_text("\n".join(kept) + "\n", encoding="utf-8")

    # csv: same, preserving column order
    idx = pd.read_csv(csv_p)
    cols = list(idx.columns)
    idx = idx[~idx["source_file"].isin(targets)]
    if new_chunks:
        add = pd.DataFrame([_index_row(c.to_dict()) for c in new_chunks])
        idx = pd.concat([idx, add[cols]], ignore_index=True)
    idx.to_csv(csv_p, index=False)

    dups = idx["chunk_id"].duplicated().sum()
    assert dups == 0, f"{dups} duplicate chunk_ids after splice"
    assert len(kept) == len(idx), f"jsonl/csv row mismatch: {len(kept)} vs {len(idx)}"

    return {
        "added_chunks": len(new_chunks),
        "source_files": len(targets),
        "jsonl_rows": len(kept),
        "csv_rows": len(idx),
    }
