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

## Local RAG Workflow

The repo now also includes a local end-to-end RAG flow for ESG PDFs:

1. extract text from one or more PDF reports;
2. chunk the text into roughly 300 to 500 token sections;
3. embed the chunks with Albert's embeddings endpoint, preferring a `BGE-M3` model when available;
4. store vectors in FAISS when installed, or fall back to local NumPy similarity search;
5. retrieve relevant chunks for a question;
6. send the retrieved context to Albert chat completions for a grounded answer.

Build an index from the bundled sample report:

```bash
python3 -m app.cli rag-build
```

Build from specific PDFs:

```bash
python3 -m app.cli rag-build \
  --pdf /absolute/path/to/report.pdf \
  --pdf /absolute/path/to/another-report.pdf
```

Preview extraction and chunking without calling Albert:

```bash
python3 -m app.cli rag-build --dry-run
```

Ask a grounded question once the index is built:

```bash
python3 -m app.cli rag-ask "What climate targets are disclosed for 2030?"
```

Inspect retrieval only:

```bash
python3 -m app.cli rag-ask "What climate targets are disclosed for 2030?" --search-only
```

The RAG commands require `ALBERT_API_KEY` for embeddings and answer generation unless you use `--dry-run`.

### Launch the visual interface

Start the local browser UI:

```bash
python3 -m app.cli rag-web
```

Then open [http://127.0.0.1:8787](http://127.0.0.1:8787).

The UI lets you:

- build an index from one or more absolute PDF paths;
- inspect the current vector backend and embedding model;
- ask ESG questions against the built index;
- review the retrieved chunks that supported the answer.

### Evaluate Retrieval

Run retrieval evaluation against the bundled ESG QA dataset:

```bash
python3 -m app.cli rag-eval
```

Evaluate a specific company explicitly:

```bash
python3 -m app.cli rag-eval --company TotalEnergies
```

The command saves JSON and CSV outputs under `outputs/` and reports which questions were clear hits, partial matches, or misses based on the expected supporting context in the dataset.

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
