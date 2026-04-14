# ESG Document Acquisition MVP

This project fetches sustainability-related corporate reports with a production-oriented retrieval pipeline designed for heterogeneous issuer websites, inconsistent report naming, redirects, mixed file formats, and partial failures.

## What It Does

For a given company, the pipeline:

1. resolves likely canonical issuer domains;
2. discovers sustainability-report candidates from structured and issuer sources;
3. ranks candidates with deterministic heuristics;
4. downloads the best raw file;
5. extracts basic text and metadata;
6. stores normalized metadata locally;
7. can run a live smoke test on the current top European listed companies by market cap.

## Retrieval Strategy

The MVP uses a tiered strategy:

- Tier 1: structured sources
  - current implementation uses a live CompaniesMarketCap adapter for dynamic ranking seeds and company profile hints
- Tier 2: issuer sources
  - issuer homepages, sustainability, ESG, investor-relations, publications, and reports pages
- Tier 3: targeted search fallback
  - DuckDuckGo HTML search using constrained issuer-domain queries

The design is intentionally modular so the ranking source, structured-source adapters, and heuristics can be extended without rewriting the whole pipeline.

## Project Structure

```text
app/
  cli.py
  models.py
  utils.py
  company_resolver.py
  source_discovery.py
  candidate_ranker.py
  downloader.py
  normalizer.py
  storage.py
  pipeline.py
  smoke_test.py
  ranking_sources.py
  parsers/
    pdf_parser.py
    html_parser.py
tests/
requirements.txt
```

## Requirements

- Python 3.11+
- network access for live discovery and the smoke test

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Run all commands from the repo root:

```bash
cd /Users/ems/Desktop/ESG_AI/repo
```

### Discover candidates for one company

```bash
python3 -m app.cli discover --company "ASML"
```

You can help resolution by passing known fields:

```bash
python3 -m app.cli discover --company "ASML" --ticker ASML --country Netherlands --issuer-domain asml.com
```

### Fetch the best report for one company

```bash
python3 -m app.cli fetch --company "ASML"
```

The command saves:

- raw files under `data/raw/<company>/`
- extracted text under `data/text/<company>/`
- candidate logs under `data/candidates/`
- metadata in `data/metadata.db`

### Run the live smoke test

```bash
python3 -m app.cli smoke-test
python3 -m app.cli smoke-test --top-n 10
```

The smoke test:

- fetches the top European listed companies dynamically at runtime from a live ranking page;
- resolves issuer sources for each company;
- tries to retrieve the latest sustainability-related report;
- stores normalized outputs locally;
- writes a compact summary to `outputs/` as CSV and JSON.

## Normalized Document Record

Each fetched document is normalized into this shape:

```json
{
  "company_name": "...",
  "ticker": "...",
  "country": "...",
  "report_year": 2025,
  "document_type": "sustainability_report",
  "title": "...",
  "source_url": "...",
  "final_url": "...",
  "source_type": "issuer_pdf",
  "mime_type": "application/pdf",
  "download_path": "...",
  "text_path": "...",
  "file_hash": "...",
  "published_date": "...",
  "discovery_confidence": 68.0,
  "parse_status": "success",
  "notes": "..."
}
```

## Candidate Ranking Heuristics

The candidate ranker is deterministic and scores:

- positive signals
  - `sustainability report`, `esg report`, `climate report`, `annual report`, `gri index`
  - recent years in the URL, title, or nearby text
  - PDF links
  - path hints like `/sustainability/`, `/esg/`, `/investor/`, `/reports/`
- negative signals
  - `press release`, `news`, `blog`, `careers`, `events`, `webcast`
  - missing year
  - low company-name similarity

The pipeline always stores a scored list rather than only the final winner.

## Tests

Run the unit tests with:

```bash
python3 -m pytest
```

Current tests cover:

- ranking preference for recent sustainability PDFs
- URL normalization and relative-link resolution

## Storage Layout

```text
data/
  metadata.db
  raw/
  text/
  candidates/
outputs/
  smoke_test_*.json
  smoke_test_*.csv
```

## Current Limitations

This MVP is deliberately conservative:

- it does not perform OCR for scanned PDFs;
- it does not use LLMs or semantic matching;
- Playwright is not the default path and is not yet wired into the main loop;
- some issuer websites require stronger anti-bot handling or region-specific headers;
- multilingual labels are only handled through obvious keyword overlap for now;
- the ranking-source adapter currently depends on CompaniesMarketCap page structure.

## Likely Extensions

The next places to extend heuristics are:

- structured adapters for official filing portals or exchange-hosted report directories;
- a stronger resolver for investor-relations subdomains and corporate website detection;
- Playwright fallback for JS-heavy report hubs;
- richer date extraction from landing pages and PDF metadata;
- duplicate clustering across annual-report, ESG-report, and sustainability-statement variants.
