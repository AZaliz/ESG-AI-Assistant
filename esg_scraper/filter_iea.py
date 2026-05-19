"""Drop IEA NZE chapters 4-10 (page > 60) from the chunk outputs.

Per scraper/extractors/eurlex.py: only the IEA Net Zero executive summary
(~pp.1-60) is signal for RAG; the sectoral-modeling chapters are retrieval
noise. This filters chunks.jsonl AND chunks_index.csv consistently, after
backing both up to *.bak (reversible).

Run from esg_scraper/:  python filter_iea.py
"""
import json
import shutil
from pathlib import Path

import pandas as pd

D = Path("data/chunks")
IEA = "2021_regulation_07c11737.pdf"
CUTOFF = 60


def drop(rec: dict) -> bool:
    """True = this chunk should be removed."""
    return rec.get("source_file") == IEA and (rec.get("page") or 0) > CUTOFF


def main() -> int:
    for f in ("chunks.jsonl", "chunks_index.csv"):
        shutil.copy(D / f, D / (f + ".bak"))

    kept = []
    dropped = 0
    with open(D / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if drop(json.loads(line)):
                dropped += 1
            else:
                kept.append(line)
    with open(D / "chunks.jsonl", "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept) + "\n")

    df = pd.read_csv(D / "chunks_index.csv")
    before = len(df)
    df = df[~((df["source_file"] == IEA) & (df["page"] > CUTOFF))]
    df.to_csv(D / "chunks_index.csv", index=False)

    print(f"chunks.jsonl     : dropped {dropped}, kept {len(kept)}")
    print(f"chunks_index.csv : {before} -> {len(df)} rows")
    print(f"backups          : {D}/chunks.jsonl.bak, {D}/chunks_index.csv.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
