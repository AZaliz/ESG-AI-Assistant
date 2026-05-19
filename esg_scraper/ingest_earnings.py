"""Ingest earnings-call transcripts into the chunk corpus.

Same treatment as the other documents: parse -> chunk with the existing
narrative chunker -> splice into data/chunks/{chunks.jsonl, chunks_index.csv}.
Transcripts are pure speaker-tagged prose (no tables), so there is no table
path — they are the .txt analogue of a narrative_only PDF.

Metadata:
  company           -> mapped from the folder slug to the corpus display name
                       (same entities already in the corpus)
  doc_type          -> "earnings_call"
  fiscal_year       -> year named in the `period` column ("FY 2022" -> 2022)
  publication_year  -> year the call was held (from `call_date`)
  source_authority  -> 2 (first-party company communication, but not the
                       audited ESRS/CSRD disclosure -> "supplement" tier)

Idempotent: re-running replaces these source_files rather than duplicating.

Usage:  python ingest_earnings.py [--dry-run]
"""
import re
import sys
from pathlib import Path

import pandas as pd

from parser.chunker import Chunk, _chunk_narrative
from parser.corpus_io import splice_chunks
from parser.taxonomy import classify_esrs_topics, speaker_role_for_turn

EC_DIR = Path("data/earnings_calls")
CHUNK_DIR = Path("data/chunks")
SOURCE_AUTHORITY = 2

SLUG_TO_COMPANY = {
    "airbus": "Airbus",
    "bnp_paribas": "BNP Paribas",
    "danone": "Danone",
    "enel": "Enel",
    "engie": "Engie",
    "l_oreal": "L'Oréal",
    "schneider_electric": "Schneider Electric",
    "siemens": "Siemens",
    "totalenergies": "TotalEnergies",
    "volkswagen": "Volkswagen",
}

_DIVIDER = re.compile(r"^=+\s*$")
_YEAR = re.compile(r"(19|20)\d{2}")

# A speaker-turn header is its OWN line: "Name  [Role]" with a known role.
# Whitelisting roles avoids matching inline annotations like
# "[indiscernible]" or "[Operator Instructions]".
_ROLES = ("Executives", "Analysts", "Operator", "Attendees")
_SPEAKER_HDR = re.compile(
    r"^(?P<name>[^\[\]]{1,60}?)\s+\[(?P<role>" + "|".join(_ROLES) + r")\]$"
)


def _split_turns(body: str) -> list[tuple[str, str, str]]:
    """Split a transcript body into (speaker, role, text) turns.

    Any text before the first header is bucketed as ('Unknown','').
    """
    turns: list[tuple[str, str, str]] = []
    cur_name, cur_role, buf = "Unknown", "", []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            turns.append((cur_name, cur_role, text))

    for ln in body.splitlines():
        m = _SPEAKER_HDR.match(ln.strip())
        if m:
            flush()
            cur_name = m.group("name").strip() or "Unknown"
            cur_role = m.group("role")
            buf = []
        else:
            buf.append(ln)
    flush()
    return turns


def _doc_id_base(source_file: str) -> str:
    # Mirror chunker.build_chunks_for_document exactly so ids stay consistent.
    return re.sub(r"[^a-z0-9]+", "_", source_file.lower())[:40]


def _speaker_chunks(body: str, doc_meta: dict) -> list[Chunk]:
    """One or more narrative chunks per speaker turn, EACH prefixed with the
    speaker+role so attribution survives the ~400-token windowing."""
    base = _doc_id_base(doc_meta["source_file"])
    chunks: list[Chunk] = []
    counter = 0
    turns = _split_turns(body) or [("Unknown", "", body)]
    for name, role, text in turns:
        tag = f"{name} [{role}]: " if role else f"{name}: "
        speaker = speaker_role_for_turn(role)
        for window in _chunk_narrative(text, page=1):
            counter += 1
            # Operator turns are call-mechanics boilerplate ("Your next
            # question comes from...") — skip emitting them. The counter is
            # still advanced so surviving chunk_ids stay byte-identical to
            # the pre-filter numbering (gaps where operators were), keeping
            # the build reproducible against the committed corpus. Chunk
            # construction + ESRS classification are skipped (no wasted work).
            if speaker == "operator":
                continue
            chunk_text = tag + window
            chunks.append(Chunk(
                chunk_id=f"{base}_n{counter:04d}",
                text=chunk_text,
                company=doc_meta["company"],
                doc_type=doc_meta["doc_type"],
                fiscal_year=doc_meta["fiscal_year"],
                publication_year=doc_meta["publication_year"],
                page=1,
                source_file=doc_meta["source_file"],
                chunk_kind="narrative",
                source_authority=doc_meta["source_authority"],
                speaker_role=speaker,
                esrs_topic=classify_esrs_topics(chunk_text, "narrative"),
            ))
    return chunks


def _strip_header(raw: str) -> str:
    """Drop the Title/Event/Source metadata block above the ==== divider."""
    lines = raw.splitlines()
    for i, ln in enumerate(lines):
        if _DIVIDER.match(ln.strip()):
            return "\n".join(lines[i + 1:]).strip()
    return raw.strip()  # no divider -> keep everything


def _year(*candidates) -> int | None:
    for c in candidates:
        if c is None or (isinstance(c, float) and pd.isna(c)):
            continue
        m = _YEAR.search(str(c))
        if m:
            return int(m.group(0))
    return None


def build() -> tuple[list, list[str]]:
    man = pd.read_csv(EC_DIR / "manifest.csv")
    all_chunks, source_files, skipped = [], [], []

    for _, r in man.iterrows():
        slug = r["company"]
        company = SLUG_TO_COMPANY.get(slug)
        if not company:
            skipped.append(f"unknown slug {slug!r}")
            continue
        path = EC_DIR / slug / str(r["year_folder"]) / r["filename"]
        if not path.exists():
            skipped.append(f"missing {path}")
            continue

        body = _strip_header(path.read_text(encoding="utf-8", errors="replace"))
        if not body.strip():
            skipped.append(f"empty body {path.name}")
            continue

        fy = _year(r.get("period"), r["filename"], r.get("call_date"))
        py = _year(r.get("call_date"), r["filename"])
        doc_meta = {
            "company": company,
            "doc_type": "earnings_call",
            "fiscal_year": fy,
            "publication_year": py,
            "source_file": r["filename"],
            "source_authority": SOURCE_AUTHORITY,
        }
        chunks = _speaker_chunks(body, doc_meta)
        all_chunks.extend(chunks)
        source_files.append(r["filename"])

    if skipped:
        print(f"  skipped {len(skipped)}: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return all_chunks, source_files


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    chunks, sfs = build()
    print(f"{len(sfs)} transcripts -> {len(chunks)} chunks")
    if dry:
        print("(dry run — corpus not modified)")
    else:
        stats = splice_chunks(chunks, sfs, chunk_dir=CHUNK_DIR,
                              backup_suffix=".earnings.bak")
        print(f"spliced: chunks.jsonl -> {stats['jsonl_rows']} rows, "
              f"chunks_index.csv -> {stats['csv_rows']} rows")
