# Known Issues

## DtypeWarning in corpus_io.splice_chunks
`corpus_io.splice_chunks` triggers a pandas DtypeWarning on the
`speaker_role` column because the column is ~96% empty (only earnings
chunks have a value). Cosmetic only — does not affect data integrity
since chunks.jsonl is the source of truth and is unaffected.
Fix when convenient: pass `dtype=str` or `low_memory=False` to
`pd.read_csv` at corpus_io.py line 68.
