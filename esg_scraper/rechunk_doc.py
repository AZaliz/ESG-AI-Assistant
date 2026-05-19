"""Targeted re-chunk: rebuild chunks for specific documents WITHOUT touching
the other ~50. Splices into the existing data/chunks/{chunks.jsonl,
chunks_index.csv} (backed up to *.bak), replacing only the matched
source_file(s). This avoids build_index's whole-file overwrite.

Usage:
    python rechunk_doc.py <source_file.pdf> [<source_file2> ...]

source_file = the basename as it appears in the manifest `file_path`
(equivalently the chunk index `source_file` column).
"""
import sys
from pathlib import Path

import pandas as pd

from parser.build_index import process_one_document
from parser.corpus_io import splice_chunks

D = Path("data/chunks")
MANIFEST = Path("data/manifest.csv")


def rechunk(source_files: list[str]) -> int:
    man = pd.read_csv(MANIFEST)
    man["__name"] = man["file_path"].apply(lambda p: Path(str(p)).name)

    new_chunks = []
    for sf in source_files:
        rows = man[man["__name"] == sf]
        if rows.empty:
            print(f"  !! no manifest row for {sf}")
            return 1
        chunks, report = process_one_document(rows.iloc[0])
        print(f"  {sf}: {report.get('narrative_chunks', 0)} narrative + "
              f"{report.get('table_fact_chunks', 0)} table_fact "
              f"= {report.get('total_chunks', len(chunks))} chunks")
        new_chunks.extend(chunks)

    stats = splice_chunks(new_chunks, source_files, chunk_dir=D,
                          backup_suffix=".rechunk.bak")
    print(f"spliced: chunks.jsonl -> {stats['jsonl_rows']} rows, "
          f"chunks_index.csv -> {stats['csv_rows']} rows")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(rechunk(sys.argv[1:]))
