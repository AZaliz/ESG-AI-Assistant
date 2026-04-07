#!/usr/bin/env python3
"""Ingest ESG disclosure PDFs into Albert and run native RAG analysis."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_COLLECTION_NAME = "esg-disclosures"
DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_QUESTIONS_FILE = DEFAULT_SAMPLE_DIR / "csrd_template_questions.yaml"
DEFAULT_COMPANY_LIST_FILE = DEFAULT_SAMPLE_DIR / "company_pdf_list.yaml"
DEFAULT_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class AlbertClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 120

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models").json().get("data", [])

    def get_text_generation_model(self) -> str:
        models = self.list_models()
        for model in models:
            if model.get("type") == "text-generation":
                return str(model["id"])
        raise RuntimeError("No Albert text-generation model was returned by /models.")

    def find_collection_by_name(self, name: str) -> dict[str, Any] | None:
        response = self._request("GET", "/collections", params={"name": name}).json()
        collections = response.get("data", [])
        for collection in collections:
            if collection.get("name") == name:
                return collection
        return None

    def create_collection(self, name: str, description: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "visibility": "private"}
        if description:
            payload["description"] = description
        return self._request("POST", "/collections", json=payload).json()

    def ensure_collection(self, name: str, description: str | None = None) -> dict[str, Any]:
        collection = self.find_collection_by_name(name)
        if collection:
            return collection
        return self.create_collection(name=name, description=description)

    def list_documents(self, collection_id: int | str) -> list[dict[str, Any]]:
        response = self._request("GET", "/documents", params={"collection_id": collection_id}).json()
        return response.get("data", [])

    def upload_document(
        self,
        file_path: Path,
        collection_id: int | str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 2048,
        chunk_overlap: int = 200,
    ) -> dict[str, Any]:
        with file_path.open("rb") as file_handle:
            files = {
                "file": (file_path.name, file_handle, "application/pdf"),
            }
            data = {
                "collection_id": str(collection_id),
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap),
            }
            if metadata:
                data["metadata"] = json.dumps(metadata)
            return self._request("POST", "/documents", files=files, data=data).json()

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        collection_id: int | str,
        limit: int = 8,
        method: str = "hybrid",
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "tools": [
                {
                    "type": "search",
                    "collection_ids": [int(collection_id)],
                    "method": method,
                    "limit": limit,
                }
            ],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        return self._request("POST", "/chat/completions", json=payload).json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload ESG disclosure PDFs to Albert and run RAG analysis."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory containing local PDF disclosures.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Albert collection name to create or reuse.",
    )
    parser.add_argument(
        "--collection-description",
        default="ESG disclosure reports indexed for RAG analysis.",
        help="Description used when a new collection is created.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help="YAML-like file containing CSRD template questions.",
    )
    parser.add_argument(
        "--company-list-file",
        type=Path,
        default=DEFAULT_COMPANY_LIST_FILE,
        help="YAML-like file listing remote company PDFs to optionally download.",
    )
    parser.add_argument(
        "--download-listed-pdfs",
        action="store_true",
        help="Download the company PDFs listed in company_pdf_list.yaml before upload.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "downloaded_reports",
        help="Where remote company PDFs should be stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "esg_analysis.json",
        help="Where to save the analysis JSON output.",
    )
    parser.add_argument(
        "--analysis-prompt",
        default=(
            "Using only evidence retrieved from the indexed ESG disclosure documents, "
            "answer the CSRD-style questions and return strict JSON with keys "
            "summary, answers, and notable_gaps. Each answer item must contain "
            "question, answer, evidence, and confidence."
        ),
        help="Prompt sent to Albert for the RAG analysis.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size used during document ingestion.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap used during document ingestion.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=8,
        help="How many search results Albert can retrieve during generation.",
    )
    parser.add_argument(
        "--search-method",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Albert SearchTool retrieval method.",
    )
    parser.add_argument(
        "--strict-downloads",
        action="store_true",
        help="Stop the run if any remote company PDF cannot be downloaded.",
    )
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.environ.get("ALBERT_API_KEY")
    if not api_key:
        raise RuntimeError("Set ALBERT_API_KEY in your environment before running this script.")
    return api_key


def read_csrd_questions(path: Path) -> list[str]:
    questions: list[str] = []
    if not path.exists():
        return questions

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- "):
            questions.append(stripped[2:].strip())
    return questions


def read_company_pdf_list(path: Path) -> list[dict[str, str]]:
    companies: list[dict[str, str]] = []
    if not path.exists():
        return companies

    current: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- name:"):
            if current:
                companies.append(current)
            current = {"name": stripped.split(":", 1)[1].strip()}
        elif stripped.startswith("name:"):
            current["name"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("pdf_url:"):
            current["pdf_url"] = stripped.split(":", 1)[1].strip()

    if current:
        companies.append(current)
    return [company for company in companies if company.get("name") and company.get("pdf_url")]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, headers=DEFAULT_DOWNLOAD_HEADERS) as response:
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_handle.write(chunk)
    return destination


def gather_pdf_files(
    sample_dir: Path,
    download_dir: Path,
    company_list_file: Path,
    download: bool,
    strict_downloads: bool,
) -> list[Path]:
    pdf_files = sorted(sample_dir.glob("*.pdf"))
    if not download:
        return pdf_files

    for company in read_company_pdf_list(company_list_file):
        target = download_dir / f"{slugify(company['name'])}.pdf"
        if not target.exists():
            print(f"Downloading {company['name']} -> {target}")
            try:
                download_file(company["pdf_url"], target)
            except requests.RequestException as exc:
                if strict_downloads:
                    raise
                print(f"Warning: skipping {company['name']} because download failed: {exc}")
                continue
        pdf_files.append(target)

    unique_paths = sorted({path.resolve() for path in pdf_files})
    return [Path(path) for path in unique_paths]


def upload_missing_documents(
    client: AlbertClient,
    collection_id: int | str,
    pdf_files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    existing_documents = client.list_documents(collection_id)
    existing_names = {doc.get("name") for doc in existing_documents}
    uploaded: list[dict[str, Any]] = []

    for pdf_file in pdf_files:
        if pdf_file.name in existing_names:
            print(f"Skipping existing document: {pdf_file.name}")
            continue

        print(f"Uploading {pdf_file.name}")
        document = client.upload_document(
            pdf_file,
            collection_id=collection_id,
            metadata={"source_file": pdf_file.name},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        uploaded.append(document)

    return uploaded


def build_messages(analysis_prompt: str, questions: list[str]) -> list[dict[str, str]]:
    question_block = "\n".join(f"- {question}" for question in questions)
    user_prompt = analysis_prompt
    if question_block:
        user_prompt += f"\n\nQuestions to cover:\n{question_block}"

    return [
        {
            "role": "system",
            "content": (
                "You are an ESG analyst. Use only retrieved evidence from Albert SearchTool. "
                "If evidence is insufficient, say so clearly in the JSON output."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def extract_message_content(chat_response: dict[str, Any]) -> str:
    choices = chat_response.get("choices") or []
    if not choices:
        raise RuntimeError("Albert returned no choices in chat completion response.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(text_parts).strip()
    raise RuntimeError("Unable to extract assistant content from chat completion response.")


def main() -> int:
    args = parse_args()
    api_key = require_api_key()

    client = AlbertClient(
        api_key=api_key,
        base_url=os.environ.get("ALBERT_BASE_URL", DEFAULT_BASE_URL),
    )

    pdf_files = gather_pdf_files(
        sample_dir=args.sample_dir,
        download_dir=args.download_dir,
        company_list_file=args.company_list_file,
        download=args.download_listed_pdfs,
        strict_downloads=args.strict_downloads,
    )
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {args.sample_dir}")

    questions = read_csrd_questions(args.questions_file)
    collection = client.ensure_collection(
        name=args.collection_name,
        description=args.collection_description,
    )
    collection_id = collection["id"]
    model = client.get_text_generation_model()

    upload_missing_documents(
        client=client,
        collection_id=collection_id,
        pdf_files=pdf_files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    messages = build_messages(args.analysis_prompt, questions)
    response = client.chat_completion(
        model=model,
        messages=messages,
        collection_id=collection_id,
        limit=args.search_limit,
        method=args.search_method,
    )
    message_content = extract_message_content(response)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "collection": {
            "id": collection_id,
            "name": collection["name"],
        },
        "documents_indexed": [pdf.name for pdf in pdf_files],
        "questions": questions,
        "raw_response": response,
        "analysis": safe_json_loads(message_content),
    }
    args.output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved analysis to {args.output}")
    print(json.dumps(output_payload["analysis"], indent=2, ensure_ascii=False))
    return 0


def safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"unparsed_response": value}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
