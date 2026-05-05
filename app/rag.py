"""Local PDF-to-RAG pipeline using Albert embeddings and optional FAISS storage."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import requests

from app.utils import OUTPUT_DIR, ensure_directories, normalize_whitespace

DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_INDEX_DIR = OUTPUT_DIR / "rag_index"
DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DEFAULT_SAMPLE_PDF = DEFAULT_SAMPLE_DIR / "totalenergies_sustainability-climate-2024-progress-report_2024_en_pdf.pdf"
TOKEN_PATTERN = re.compile(r"\S+")

try:  # pragma: no cover - availability depends on the local environment.
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback path is covered instead.
    faiss = None


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    source_path: str
    page_start: int
    page_end: int
    token_count: int
    text: str


class AlbertClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 120) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models").json().get("data", [])

    def get_embedding_model(self, preferred: str | None = "bge-m3") -> str:
        models = self.list_models()
        if not models:
            raise RuntimeError("Albert returned no models.")

        if preferred:
            preferred_lower = preferred.lower()
            for model in models:
                identifier = str(model.get("id", ""))
                if preferred_lower in identifier.lower():
                    return identifier

        embedding_types = {"embedding", "embeddings", "text-embedding", "text-embeddings"}
        for model in models:
            model_type = str(model.get("type", "")).lower()
            if model_type in embedding_types:
                return str(model["id"])

        for model in models:
            identifier = str(model.get("id", ""))
            if "embed" in identifier.lower():
                return identifier

        raise RuntimeError("No Albert embedding model was returned by /models.")

    def get_text_generation_model(self) -> str:
        for model in self.list_models():
            if str(model.get("type", "")).lower() == "text-generation":
                return str(model["id"])
        raise RuntimeError("No Albert text-generation model was returned by /models.")

    def create_embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        payload = {"model": model, "input": inputs}
        response = self._request("POST", "/embeddings", json=payload).json()
        data = response.get("data") or []
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in ordered]
        if not embeddings or any(not isinstance(embedding, list) for embedding in embeddings):
            raise RuntimeError("Albert returned an unexpected embeddings payload.")
        return embeddings

    def chat_completion(self, model: str, messages: list[dict[str, str]]) -> str:
        response = self._request(
            "POST",
            "/chat/completions",
            json={"model": model, "messages": messages, "stream": False},
        ).json()
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Albert returned no chat completion choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in text_parts if part).strip()
        raise RuntimeError("Albert returned an unexpected chat completion payload.")


def require_api_key() -> str:
    api_key = os.environ.get("ALBERT_API_KEY")
    if not api_key:
        raise RuntimeError("Set ALBERT_API_KEY in your environment before running the RAG commands.")
    return api_key


def set_api_key(api_key: str | None) -> None:
    if api_key:
        os.environ["ALBERT_API_KEY"] = api_key.strip()


def estimate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    words = TOKEN_PATTERN.findall(text)
    if len(words) <= max_tokens:
        return [normalize_whitespace(text)]

    window_count = (len(words) + max_tokens - 1) // max_tokens
    window_size = min(max_tokens, (len(words) + window_count - 1) // window_count)
    windows: list[str] = []
    for start in range(0, len(words), window_size):
        window = words[start : start + window_size]
        if window:
            windows.append(" ".join(window))
    return windows


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            text = raw_text.strip()
            if text:
                pages.append(PageText(page_number=index, text=text))
    return pages


def _build_units(pages: list[PageText], max_tokens: int) -> list[tuple[int, str, int]]:
    units: list[tuple[int, str, int]] = []
    for page in pages:
        paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", page.text)]
        paragraphs = [part for part in paragraphs if part]
        if not paragraphs:
            paragraphs = [page.text]
        for paragraph in paragraphs:
            token_count = estimate_token_count(paragraph)
            for window in _split_long_text(paragraph, max_tokens=max_tokens):
                units.append((page.page_number, window, estimate_token_count(window)))
    return units


def _finalize_chunk(units: list[tuple[int, str, int]], chunk_number: int, source_file: str, source_path: str) -> ChunkRecord:
    chunk_text = "\n\n".join(text for _, text, _ in units).strip()
    page_numbers = [page_number for page_number, _, _ in units]
    return ChunkRecord(
        chunk_id=f"chunk-{chunk_number:04d}",
        source_file=source_file,
        source_path=source_path,
        page_start=min(page_numbers),
        page_end=max(page_numbers),
        token_count=estimate_token_count(chunk_text),
        text=chunk_text,
    )


def _merge_chunk_records(first: ChunkRecord, second: ChunkRecord, chunk_id: str) -> ChunkRecord:
    merged_text = f"{first.text}\n\n{second.text}".strip()
    return ChunkRecord(
        chunk_id=chunk_id,
        source_file=first.source_file,
        source_path=first.source_path,
        page_start=min(first.page_start, second.page_start),
        page_end=max(first.page_end, second.page_end),
        token_count=estimate_token_count(merged_text),
        text=merged_text,
    )


def _compact_small_chunks(chunks: list[ChunkRecord], min_tokens: int, max_tokens: int) -> list[ChunkRecord]:
    compacted: list[ChunkRecord] = []
    soft_max_tokens = max_tokens + 50
    index = 0
    while index < len(chunks):
        current = chunks[index]

        while (
            current.token_count < min_tokens
            and index + 1 < len(chunks)
            and chunks[index + 1].source_file == current.source_file
        ):
            candidate = chunks[index + 1]
            combined = estimate_token_count(f"{current.text}\n\n{candidate.text}")
            if combined <= max_tokens or (
                current.token_count < (min_tokens // 2) and combined <= soft_max_tokens
            ):
                current = _merge_chunk_records(current, candidate, chunk_id=current.chunk_id)
                index += 1
                continue
            break

        if (
            compacted
            and current.token_count < min_tokens
            and compacted[-1].source_file == current.source_file
        ):
            combined = estimate_token_count(f"{compacted[-1].text}\n\n{current.text}")
            if combined <= max_tokens or (
                current.token_count < (min_tokens // 2) and combined <= soft_max_tokens
            ):
                previous = compacted.pop()
                current = _merge_chunk_records(previous, current, chunk_id=previous.chunk_id)

        compacted.append(current)
        index += 1

    for chunk_number, chunk in enumerate(compacted, start=1):
        chunk.chunk_id = f"chunk-{chunk_number:04d}"
    return compacted


def _tail_units_by_tokens(units: list[tuple[int, str, int]], overlap_tokens: int) -> list[tuple[int, str, int]]:
    if overlap_tokens <= 0:
        return []

    collected: list[tuple[int, str, int]] = []
    running_total = 0
    for unit in reversed(units):
        collected.append(unit)
        running_total += unit[2]
        if running_total >= overlap_tokens:
            break
    return list(reversed(collected))


def chunk_pages(
    pages: list[PageText],
    *,
    source_file: str,
    source_path: str,
    target_tokens: int = 420,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
) -> list[ChunkRecord]:
    units = _build_units(pages, max_tokens=max_tokens)
    if not units:
        return []

    chunks: list[ChunkRecord] = []
    current_units: list[tuple[int, str, int]] = []
    current_tokens = 0

    for unit in units:
        page_number, text, token_count = unit
        if not current_units:
            current_units.append((page_number, text, token_count))
            current_tokens = token_count
            continue

        should_append = current_tokens + token_count <= max_tokens and (
            current_tokens < target_tokens or current_tokens < min_tokens
        )
        if should_append:
            current_units.append((page_number, text, token_count))
            current_tokens += token_count
            continue

        chunks.append(
            _finalize_chunk(
                current_units,
                chunk_number=len(chunks) + 1,
                source_file=source_file,
                source_path=source_path,
            )
        )
        overlap_units = _tail_units_by_tokens(current_units, overlap_tokens)
        if sum(unit[2] for unit in overlap_units) >= max_tokens:
            overlap_units = [current_units[-1]]
        current_units = [*overlap_units, (page_number, text, token_count)]
        current_tokens = sum(unit[2] for unit in current_units)

    if current_units:
        chunks.append(
            _finalize_chunk(
                current_units,
                chunk_number=len(chunks) + 1,
                source_file=source_file,
                source_path=source_path,
            )
        )

    return chunks


def chunk_text(
    text: str,
    *,
    source_file: str = "document.txt",
    source_path: str = "document.txt",
    target_tokens: int = 420,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
) -> list[ChunkRecord]:
    pages = [PageText(page_number=1, text=text)]
    return chunk_pages(
        pages,
        source_file=source_file,
        source_path=source_path,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def gather_pdf_paths(pdf_paths: list[str] | None) -> list[Path]:
    if pdf_paths:
        resolved = [Path(path).expanduser().resolve() for path in pdf_paths]
        return resolved

    if DEFAULT_SAMPLE_PDF.exists():
        return [DEFAULT_SAMPLE_PDF.resolve()]

    default_pdfs = sorted(DEFAULT_SAMPLE_DIR.glob("*.pdf"))
    return [path.resolve() for path in default_pdfs]


def parse_pdf_path_lines(raw_value: str) -> list[Path]:
    values = [line.strip() for line in raw_value.splitlines() if line.strip()]
    return gather_pdf_paths(values or None)


def get_index_summary(index_dir: Path) -> dict[str, Any] | None:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_chunk_records(
    pdf_paths: list[Path],
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[ChunkRecord]:
    chunk_records: list[ChunkRecord] = []
    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        chunk_records.extend(
            chunk_pages(
                pages,
                source_file=pdf_path.name,
                source_path=str(pdf_path),
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )

    return _compact_small_chunks(chunk_records, min_tokens=min_tokens, max_tokens=max_tokens)


def _normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(
    client: AlbertClient,
    texts: list[str],
    *,
    embedding_model: str,
    batch_size: int = 16,
) -> np.ndarray:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(client.create_embeddings(embedding_model, batch))
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.size == 0:
        raise RuntimeError("No embeddings were generated for the chunks.")
    return _normalize_embeddings(vectors)


def search_vectors(query_vector: np.ndarray, vectors: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    if vectors.ndim != 2 or vectors.size == 0:
        return []
    normalized_query = _normalize_embeddings(query_vector.reshape(1, -1))[0]
    normalized_vectors = _normalize_embeddings(vectors)
    scores = normalized_vectors @ normalized_query
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(index), float(scores[index])) for index in top_indices]


def _load_chunks(index_dir: Path) -> list[ChunkRecord]:
    payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    return [ChunkRecord(**item) for item in payload]


def _store_index(index_dir: Path, vectors: np.ndarray) -> str:
    np.save(index_dir / "vectors.npy", vectors)
    if faiss is None:
        return "numpy"

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.copy())
    faiss.write_index(index, str(index_dir / "index.faiss"))
    return "faiss"


def _load_vector_backend(index_dir: Path, backend: str) -> Any:
    if backend == "faiss" and faiss is not None and (index_dir / "index.faiss").exists():
        return faiss.read_index(str(index_dir / "index.faiss"))
    return np.load(index_dir / "vectors.npy")


def build_index(
    *,
    pdf_paths: list[Path],
    index_dir: Path,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
    batch_size: int,
    embedding_model: str | None = None,
    dry_run: bool = False,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    ensure_directories()
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunk_records(
        pdf_paths,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    if not chunks:
        raise RuntimeError("No extractable text was found in the provided PDFs.")

    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "built_at": datetime.now(UTC).isoformat(),
        "pdfs": [str(path) for path in pdf_paths],
        "chunk_count": len(chunks),
        "target_tokens": target_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "vector_backend": "none",
    }

    if dry_run:
        (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_embedding_model = embedding_model or client.get_embedding_model(preferred="bge-m3")
    vectors = embed_texts(
        client,
        [chunk.text for chunk in chunks],
        embedding_model=selected_embedding_model,
        batch_size=batch_size,
    )
    vector_backend = _store_index(index_dir, vectors)

    manifest.update(
        {
            "embedding_model": selected_embedding_model,
            "embedding_dimension": int(vectors.shape[1]),
            "vector_backend": vector_backend,
        }
    )
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def retrieve_chunks(
    *,
    index_dir: Path,
    question: str,
    top_k: int,
    embedding_model: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[list[dict[str, Any]], str]:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    vector_backend = manifest.get("vector_backend", "none")
    if vector_backend == "none":
        raise RuntimeError("This index was built with --dry-run and has no embeddings to search.")

    chunks = _load_chunks(index_dir)
    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_embedding_model = embedding_model or manifest.get("embedding_model")
    if not selected_embedding_model:
        selected_embedding_model = client.get_embedding_model(preferred="bge-m3")

    query_vector = embed_texts(
        client,
        [question],
        embedding_model=selected_embedding_model,
        batch_size=1,
    )[0]
    store = _load_vector_backend(index_dir, vector_backend)

    if vector_backend == "faiss" and faiss is not None and hasattr(store, "search"):
        scores, indices = store.search(query_vector.reshape(1, -1), top_k)
        matches = [(int(index), float(score)) for score, index in zip(scores[0], indices[0]) if index >= 0]
    else:
        matches = search_vectors(query_vector, store, top_k)

    results: list[dict[str, Any]] = []
    for chunk_index, score in matches:
        chunk = chunks[chunk_index]
        results.append(
            {
                "score": score,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "source_path": chunk.source_path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
        )
    return results, selected_embedding_model


def answer_question(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    text_model: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[str, str]:
    if not retrieved_chunks:
        raise RuntimeError("No retrieved chunks were provided to the answer stage.")

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_text_model = text_model or client.get_text_generation_model()
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            (
                f"{chunk['chunk_id']} | {chunk['source_file']} | pages {chunk['page_start']}-{chunk['page_end']}\n"
                f"{chunk['text']}"
            )
        )
    context = "\n\n".join(context_blocks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an ESG analyst. Answer only from the supplied context. "
                "If the context is insufficient, say so clearly. Cite supporting chunk ids in brackets."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        },
    ]
    return client.chat_completion(selected_text_model, messages), selected_text_model


def run_rag_build(args: Any) -> int:
    pdf_paths = gather_pdf_paths(args.pdf)
    if not pdf_paths:
        raise RuntimeError("No PDF files were found. Pass --pdf or add a PDF under sample_data/.")

    manifest = build_index(
        pdf_paths=pdf_paths,
        index_dir=args.index_dir,
        target_tokens=args.chunk_target_tokens,
        min_tokens=args.chunk_min_tokens,
        max_tokens=args.chunk_max_tokens,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        dry_run=args.dry_run,
        base_url=args.base_url,
    )
    mode = "dry run" if args.dry_run else manifest["vector_backend"]
    print(f"Indexed {manifest['chunk_count']} chunks from {len(pdf_paths)} PDF(s) into {args.index_dir} ({mode}).")
    if not args.dry_run:
        print(f"Embedding model: {manifest['embedding_model']}")
    return 0


def run_rag_ask(args: Any) -> int:
    retrieved, embedding_model = retrieve_chunks(
        index_dir=args.index_dir,
        question=" ".join(args.question).strip(),
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        base_url=args.base_url,
    )
    print(f"Retrieved {len(retrieved)} chunks using {embedding_model}:")
    for chunk in retrieved:
        preview = chunk["text"][:220].replace("\n", " ")
        print(
            f"- {chunk['chunk_id']} | score={chunk['score']:.4f} | "
            f"{chunk['source_file']} | pages {chunk['page_start']}-{chunk['page_end']} | {preview}"
        )

    if args.search_only:
        return 0

    answer, text_model = answer_question(
        question=" ".join(args.question).strip(),
        retrieved_chunks=retrieved,
        text_model=args.text_model,
        base_url=args.base_url,
    )
    print(f"\nAnswer ({text_model}):\n{answer}")
    return 0
