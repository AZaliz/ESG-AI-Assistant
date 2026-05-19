"""Follow-up: corpus_io.splice_chunks append / idempotency / dup-guard.

Operates on a throwaway temp corpus — never touches data/chunks/.

Run: python tests/test_corpus_io.py   (or pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # esg_scraper/

import pandas as pd

from parser.chunker import Chunk
from parser.corpus_io import splice_chunks, existing_source_files


def _chunk(cid: str, src: str) -> Chunk:
    return Chunk(
        chunk_id=cid, text=f"text for {cid}", company="ACME",
        doc_type="earnings_call", fiscal_year=2024, publication_year=2024,
        page=1, source_file=src, chunk_kind="narrative", source_authority=2,
    )


def _seed(d: Path, chunks: list[Chunk]) -> None:
    (d / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8")
    rows = []
    for c in chunks:
        r = c.to_dict()
        r["text_preview"] = r.pop("text")[:120]
        rows.append(r)
    pd.DataFrame(rows).to_csv(d / "chunks_index.csv", index=False)


def test_append_then_idempotent_then_dupguard():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _seed(d, [_chunk("old_n0001", "old.pdf"), _chunk("old_n0002", "old.pdf")])

        assert existing_source_files(d) == {"old.pdf"}

        new = [_chunk("new_n0001", "new.pdf"), _chunk("new_n0002", "new.pdf"),
               _chunk("new_n0003", "new.pdf")]
        s1 = splice_chunks(new, ["new.pdf"], chunk_dir=d)
        assert s1["jsonl_rows"] == 5 and s1["csv_rows"] == 5
        assert existing_source_files(d) == {"old.pdf", "new.pdf"}

        # Idempotent: same source_file re-spliced -> drop+readd, still 5 rows.
        s2 = splice_chunks(new, ["new.pdf"], chunk_dir=d)
        assert s2["jsonl_rows"] == 5 and s2["csv_rows"] == 5
        ids = [json.loads(l)["chunk_id"]
               for l in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        assert len(ids) == len(set(ids)) == 5

        # Dup-guard: two chunks sharing a chunk_id must raise.
        bad = [_chunk("dup_x", "bad.pdf"), _chunk("dup_x", "bad.pdf")]
        raised = False
        try:
            splice_chunks(bad, ["bad.pdf"], chunk_dir=d)
        except AssertionError:
            raised = True
        assert raised, "expected AssertionError on duplicate chunk_ids"


if __name__ == "__main__":
    test_append_then_idempotent_then_dupguard()
    print("PASS test_append_then_idempotent_then_dupguard")
