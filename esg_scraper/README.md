# ESG Document Scraper

Multi-company scraper for corporate sustainability documents. Targets 10 European companies (TotalEnergies, BNP Paribas, Airbus, Danone, Engie, Schneider Electric, L'Oréal, Volkswagen, Siemens, Enel) for the FTD Master AI-Powered ESG Analysis course. (Enel replaced Iberdrola — see [Note on coverage](#note-on-coverage).)

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (uses modern type hints).

For development, install from `requirements.txt`. For exact reproduction of
the current pre-embedding corpus, install from `requirements_locked.txt`
(a `pip freeze` of the environment that built the committed `chunks.jsonl`).
Keep both — `requirements.txt` is the human-readable loose pin,
`requirements_locked.txt` is the byte-reproduction pin.

## Project structure

```
.
├── main.py                       # CLI entry point
├── verification.py               # Quality check on the manifest
├── requirements.txt
└── scraper/
    ├── __init__.py
    ├── models.py                 # Document dataclass + doc_type vocab + ESG_REPORT_DOC_TYPES
    ├── base.py                   # BaseScraper (HTTP, session warmup, dedup, manifest filter)
    ├── config.py                 # company → scraper class registry
    └── extractors/
        ├── __init__.py
        ├── totalenergies.py
        ├── engie.py
        ├── bnp_paribas.py
        ├── airbus.py
        ├── danone.py
        ├── loreal.py
        ├── volkswagen.py
        ├── siemens.py
        ├── schneider.py
        ├── enel.py
        └── eurlex.py             # EU regulations / framework standards
```

## Usage

```bash
# 1. List available companies
python main.py --list

# 2. Dry run (discover only, no downloads) — ALWAYS DO THIS FIRST
python main.py --all --dry-run

# 3. Real run — single company
python main.py --company totalenergies

# 4. Real run — multiple specific companies
python main.py --company engie --company bnp_paribas --company enel

# 5. Real run — all 10 companies, parallel
python main.py --all --workers 3

# 6. Include policy docs (codes of conduct, supplier codes, etc.)
#    Default keeps ESG reports only.
python main.py --all --with-policies

# 7. Sanity-check the manifest afterwards
python verification.py --manifest data/manifest.csv
```

Output paths are resolved relative to `main.py`, so the script can be launched
from any directory and the corpus always lands inside `esg_scraper/data/`.

## Output

```
data/
├── pdfs/
│   ├── totalenergies/
│   │   ├── 2024_urd_a3f9b2c1.pdf
│   │   ├── 2025_progress_report_8e1d4f02.pdf
│   │   └── ...
│   ├── bnp_paribas/
│   └── ...
└── manifest.csv     # one row per document, queryable from pandas
```

The manifest contains: `company, doc_type, fiscal_year, publication_year, title, url, file_path, file_hash, size_bytes, language, priority, source_hub, downloaded_at, notes`.

## Filtering for your RAG pipeline

```python
import pandas as pd

df = pd.read_csv("data/manifest.csv")

# Only primary-priority documents for the latest fiscal year
df_primary = df[(df.priority == 1) & (df.fiscal_year >= 2024)]

# Only sustainability reports (the core CSRD/ESRS docs)
df_sustainability = df[df.doc_type.isin([
    "sustainability_report", "urd", "integrated_report", "progress_report"
])]

# Group by company to build per-company corpora
for company, group in df_primary.groupby("company"):
    print(f"{company}: {len(group)} docs")
```

## Doc type vocabulary

| Type | When |
|---|---|
| `urd` | Universal Registration Document (FR companies) |
| `annual_report` | Combined annual + ESRS statement |
| `sustainability_report` | Standalone ESRS / non-financial information |
| `integrated_report` | IIRC-style integrated report |
| `climate_report` | Climate / TCFD-specific |
| `progress_report` | TotalEnergies-style annual progress |
| `esg_tracker` | Schneider quarterly SSI |
| `esg_databook` | Quantitative tables (often `.xlsx`) |
| `framework_disclosure` | CDP, GRI, SASB, WEF, TCFD voluntary disclosures |
| `thematic_report` | Single topic (palm oil, methane, biodiversity) |
| `vigilance_plan` | French Duty of Vigilance Law |
| `policy` | Code of conduct, sector policies |
| `other` | Catch-all |

## Priority levels

| Priority | Meaning | RAG indexing |
|---|---|---|
| 1 (PRIMARY) | Core ESRS / CSRD doc | Always index |
| 2 (SECONDARY) | Supplementary | Index if budget allows |
| 3 (ARCHIVE) | Background only | Don't index by default |

## When extractors break

Sites get redesigned, document types get renamed. The framework minimizes maintenance cost:

1. **Run `--dry-run` regularly.** Discovery is cheap. Diff against previous runs to spot breakage.
2. **Pattern lists are the only thing to change.** When a doc gets renamed (e.g., "Sustainability Report" → "Sustainability Statement"), add one regex line to `TITLE_PATTERNS`.
3. **Hardcoded URLs are escape hatches.** When discovery for Volkswagen, Siemens, or Schneider stops working (their UUID/year-subdomain URLs break), add the latest known PDF URL to the `HARDCODED` list. Fix discovery later.
4. **One company breaking ≠ run failing.** Each `discover()` is wrapped in try/except in the runner.

## Notes per company

Most companies have full discovery + a hardcoded fallback. Three exceptions where hardcoded URLs matter most:

- **Airbus** — ESG Datasheet uses unguessable mediassets URLs. Update HARDCODED yearly.
- **Volkswagen** — annual report uses per-year subdomains (`annualreport2024.volkswagen-group.com`). Add the new subdomain when FY2025 drops.
- **Siemens** — PDFs at UUID-based URLs that change every release. Hardcoded fallback for the current Sustainability Statement.

## Politeness & anti-bot handling

- 1.5s delay between requests
- Sends Chrome-mimic headers (User-Agent, Accept, Sec-Fetch-*, Accept-Encoding
  with `br`) plus a session-warming GET against the site root before discovery.
  This is what gets BNP Paribas and Schneider Electric past their WAFs.
- One open connection per host (parallelism is at the company level, not document level)
- Does not bulk-download; respects implicit throttling

**Requires the `brotli` package** (in `requirements.txt`). Without it, the WAFs
that check for `Accept-Encoding: br` will return 403.

## Note on coverage

Iberdrola was originally on the target list as the European utility
representative, but its corporate site blocks scraping from non-residential
IPs (Akamai WAF — it checks the connection's JA3 TLS fingerprint, not just
HTTP headers; bypassing it would require `curl_cffi` or Playwright). **Enel
(Italy) was substituted**, providing equivalent ESRS E1/E4 coverage with
stronger climate-policy-advocacy disclosure. The Iberdrola extractor is not
registered in the active scraper config and its module has been removed from
the tree; it can be re-added under `extractors/iberdrola.py` to revive it.

## Known limitations

- Schneider Electric discovers ~6 docs but the asset CDN 403s on a couple of
  PDF downloads. The critical FY2024 Sustainability Report comes through.

For the 10 reachable companies (no `--with-policies`), expect ~190 documents
totaling ~1.9 GB. Full run takes ~3 minutes on a typical residential connection.

## Reproducibility

The pre-embedding corpus (`data/chunks/chunks.jsonl`) is reproducible
from three frozen inputs:
 - the code in this repository
 - the PDF/transcript inputs in `data/pdfs/` and `data/earnings_calls/`
   (snapshot date: 2026-05-19)
 - the pinned environment in `requirements_locked.txt`

Given those inputs, the build chain is two steps (run from `esg_scraper/`):

```bash
python -m parser.build_index
python ingest_earnings.py
```

This produces `chunks.jsonl` with the same chunk_id set as the
committed corpus. Tested end-to-end on CSDDD: 0 boundary drift,
chunk_id list byte-identical pre/post rechunk.

**Caveat:** this guarantee holds for the frozen input snapshot only.
Re-scraping ESG reports from corporate websites will produce different
PDFs (companies reissue reports, restate prior-year figures, change
layouts). A naive re-scrape followed by full rebuild may produce
different chunk boundaries even with identical code. To reproduce the
current corpus exactly, use the frozen PDF set, not a fresh scrape.

## Input snapshots

The full frozen input set used to build the current corpus is not
committed to git (~2.1 GB). It is archived separately as
`data_pdfs_snapshot_2026-05-19.zip` (despite the name, this is the
complete build-input set, not only PDFs):

 - `data/pdfs/**` — 252 scraped PDFs (`parser.build_index` input)
 - `data/manifest.csv` — build driver: doc selection, metadata, and the
   CSDDD `narrative_only` URL match
 - `data/earnings_calls/**` — 100 transcripts + earnings manifest
   (`ingest_earnings.py` input)

**SHA-256:** `ab5513eb3e892250fe922ab5828a7c68d65c729b6cc8ddb3dff67c991f31a0c3`
(also recorded in the sidecar `data_pdfs_snapshot_2026-05-19.zip.sha256`).
Verify the archive before trusting a reproduction:

```bash
sha256sum -c data_pdfs_snapshot_2026-05-19.zip.sha256
```

Contact the project owner for the archive, or re-scrape using
`python main.py --all` — note that re-scraping may produce different
PDFs from the frozen snapshot (see [Reproducibility](#reproducibility)).
