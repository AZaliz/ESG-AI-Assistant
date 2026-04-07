# ESG Disclosures RAG with Albert API

This project indexes ESG disclosure PDFs into an Albert API collection and lets you ask questions about them using Albert's native RAG search tool.

## How It Works

The workflow has two steps:

1. `esg_rag_runner.py`
   - creates or reuses an Albert collection;
   - uploads the PDF files from `sample_data/`;
   - can optionally download extra company reports listed in `sample_data/company_pdf_list.yaml`;
   - runs an initial ESG analysis with Albert `chat/completions` and `SearchTool`;
   - saves the result to `outputs/esg_analysis.json`.

2. `ask_albert_esg.py`
   - sends a one-off question to Albert;
   - searches the indexed collection with native RAG;
   - returns an answer based only on the uploaded documents.

In practice, the scripts call:
- `POST /v1/collections`
- `POST /v1/documents`
- `POST /v1/chat/completions` with `tools: [{ "type": "search", ... }]`

## Project Structure

- `esg_rag_runner.py`: upload documents and run an initial analysis
- `ask_albert_esg.py`: ask one-off questions from the terminal
- `sample_data/`: local ESG PDFs and question templates
- `albert_api_docs/`: notes on Albert collections, documents, and chat completions

## Requirements

- Python 3
- `requests`
- an Albert API key

Install the dependency:

```bash
pip3 install requests
```

Set your API key:

```bash
export ALBERT_API_KEY="your_key_here"
```

## Usage

Go into the project folder:

```bash
cd /Users/ems/Desktop/ESG_AI/repo
```

### 1. Index the local ESG PDF and run an analysis

```bash
python3 esg_rag_runner.py
```

### 2. Download extra company PDFs and index them too

```bash
python3 esg_rag_runner.py --download-listed-pdfs
```

If one company PDF is blocked by the source website, the script skips it and continues.

### 3. Ask a one-off question

```bash
python3 ask_albert_esg.py "What climate-related targets are disclosed in the TotalEnergies report?"
```

### 4. Get the raw Albert JSON response

```bash
python3 ask_albert_esg.py --json "What assurance level applies to sustainability information?"
```

## Notes

- The default Albert collection name is `esg-disclosures`.
- The scripts reuse the existing collection if it already exists.
- Generated files are written to `outputs/` and are not meant to be committed.
- Broad prompts can take longer and may time out if Albert is slow.

## Create Your Own GitHub Repository

This folder was cloned from the original class repository, so its current `origin` still points there. If you want your own repo, create a new empty repository on GitHub first, then run:

```bash
cd /Users/ems/Desktop/ESG_AI/repo
git remote rename origin upstream
git remote add origin https://github.com/YOUR_USERNAME/YOUR_NEW_REPO.git
git add .
git commit -m "Add Albert ESG RAG scripts"
git push -u origin main
```

If your local branch is not `main`, check it with:

```bash
git branch --show-current
```

and push that branch name instead.
