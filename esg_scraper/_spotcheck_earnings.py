"""Throwaway: sample 3 random earnings_call chunks per company for hand audit."""
import json, random, collections, csv, pathlib

random.seed(20260519)

CH = pathlib.Path("data/chunks/chunks.jsonl")
MAN = pathlib.Path("data/earnings_calls/manifest.csv")

# manifest: filename -> (period, call_date) for fiscal-year ground truth
truth = {}
with MAN.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        truth[r["filename"]] = (r["period"], r["call_date"])

by_company = collections.defaultdict(list)
with CH.open(encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if o.get("doc_type") == "earnings_call":
            by_company[o["company"]].append(o)

print(f"companies: {len(by_company)}; total earnings chunks: {sum(len(v) for v in by_company.values())}")
for c in sorted(by_company):
    print(f"  {c}: {len(by_company[c])} chunks")

print("\n" + "=" * 100)
for company in sorted(by_company):
    picks = random.sample(by_company[company], 3)
    for o in picks:
        sf = o["source_file"]
        period, call_date = truth.get(sf, ("?? NOT IN MANIFEST", "??"))
        print(f"\n##### {company} | chunk_id={o['chunk_id']}")
        print(f"  source_file   : {sf}")
        print(f"  manifest      : period={period!r} call_date={call_date!r}")
        print(f"  fiscal_year   : {o['fiscal_year']}   publication_year: {o['publication_year']}")
        print(f"  authority={o['source_authority']} kind={o['chunk_kind']} page={o['page']} len={len(o['text'])}")
        print("  ---- TEXT ----")
        print(o["text"])
        print("  ---- /TEXT ----")
print("\n" + "=" * 100)
