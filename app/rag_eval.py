"""Retrieval evaluation helpers for the local ESG RAG pipeline."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.rag import DEFAULT_INDEX_DIR, retrieve_chunks
from app.utils import OUTPUT_DIR

DEFAULT_EVAL_DATASET = Path(__file__).resolve().parent.parent / "sample_data" / "rag_evaluation_dataset.csv"
DEFAULT_EVAL_OUTPUT = OUTPUT_DIR / "rag_eval_results.json"


def _normalize_match_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _load_rows(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_expected_contexts(row: dict[str, str]) -> list[str]:
    contexts_field = row.get("contexts", "").strip()
    if contexts_field:
        try:
            parsed = json.loads(contexts_field)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

    context = row.get("context", "").strip()
    return [context] if context else []


def _infer_company(index_dir: Path, rows: list[dict[str, str]]) -> str | None:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    pdf_blob = " ".join(manifest.get("pdfs", [])).lower()
    companies = {row["company"] for row in rows if row.get("company")}
    matches = [company for company in companies if company.lower().replace(" ", "") in pdf_blob.replace(" ", "")]
    if len(matches) == 1:
        return matches[0]

    slug_matches = []
    for company in companies:
        slug = re.sub(r"[^a-z0-9]+", "", company.lower())
        if slug and slug in re.sub(r"[^a-z0-9]+", "", pdf_blob):
            slug_matches.append(company)
    if len(slug_matches) == 1:
        return slug_matches[0]
    return None


def _score_chunk_against_snippet(snippet: str, chunk_text: str) -> float:
    normalized_snippet = _normalize_match_text(snippet)
    normalized_chunk = _normalize_match_text(chunk_text)
    if not normalized_snippet or not normalized_chunk:
        return 0.0
    if normalized_snippet in normalized_chunk:
        return 100.0
    return max(
        float(fuzz.partial_ratio(normalized_snippet, normalized_chunk)),
        float(fuzz.token_set_ratio(normalized_snippet, normalized_chunk)),
    )


def _score_retrieval(expected_contexts: list[str], retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    best_score = 0.0
    best_chunk_id = None
    best_snippet = None

    for snippet in expected_contexts:
        for chunk in retrieved_chunks:
            score = _score_chunk_against_snippet(snippet, chunk["text"])
            if score > best_score:
                best_score = score
                best_chunk_id = chunk["chunk_id"]
                best_snippet = snippet

    if best_score >= 90:
        status = "hit"
    elif best_score >= 70:
        status = "partial"
    else:
        status = "miss"

    return {
        "status": status,
        "best_match_score": round(best_score, 2),
        "best_chunk_id": best_chunk_id,
        "matched_context": best_snippet,
    }


def evaluate_retrieval(
    *,
    index_dir: Path,
    dataset_path: Path = DEFAULT_EVAL_DATASET,
    company: str | None = None,
    top_k: int = 5,
    base_url: str,
) -> dict[str, Any]:
    rows = _load_rows(dataset_path)
    selected_company = company or _infer_company(index_dir, rows)
    if not selected_company:
        raise RuntimeError("Could not infer which company to evaluate. Pass --company explicitly.")

    company_rows = [row for row in rows if row.get("company") == selected_company]
    if not company_rows:
        raise RuntimeError(f"No evaluation rows found for company '{selected_company}'.")

    results: list[dict[str, Any]] = []
    for row in company_rows:
        question = row["question"]
        expected_contexts = _parse_expected_contexts(row)
        retrieved_chunks, embedding_model = retrieve_chunks(
            index_dir=index_dir,
            question=question,
            top_k=top_k,
            base_url=base_url,
        )
        scoring = _score_retrieval(expected_contexts, retrieved_chunks)
        results.append(
            {
                "company": selected_company,
                "question": question,
                "topic": row.get("topic", ""),
                "ground_truth": row.get("ground_truth", ""),
                "status": scoring["status"],
                "best_match_score": scoring["best_match_score"],
                "best_chunk_id": scoring["best_chunk_id"],
                "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
                "top_score": round(float(retrieved_chunks[0]["score"]), 4) if retrieved_chunks else None,
                "expected_context": scoring["matched_context"],
            }
        )

    counts = Counter(result["status"] for result in results)
    topic_counts = Counter((result["topic"], result["status"]) for result in results)
    by_topic: dict[str, dict[str, int]] = {}
    for (topic, status), count in topic_counts.items():
        by_topic.setdefault(topic, {"hit": 0, "partial": 0, "miss": 0})
        by_topic[topic][status] = count

    return {
        "company": selected_company,
        "question_count": len(results),
        "top_k": top_k,
        "summary": {
            "hit": counts.get("hit", 0),
            "partial": counts.get("partial", 0),
            "miss": counts.get("miss", 0),
            "hit_rate": round(counts.get("hit", 0) / len(results), 3),
        },
        "by_topic": by_topic,
        "results": results,
        "embedding_model": embedding_model,
    }


def write_evaluation_outputs(payload: dict[str, Any], output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    rows = payload["results"]
    headers = [
        "company",
        "question",
        "topic",
        "status",
        "best_match_score",
        "best_chunk_id",
        "top_score",
        "ground_truth",
        "retrieved_chunk_ids",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    header: (" | ".join(row["retrieved_chunk_ids"]) if header == "retrieved_chunk_ids" else row.get(header, ""))
                    for header in headers
                }
            )
    return output_path, csv_path


def run_rag_eval(args: Any) -> int:
    payload = evaluate_retrieval(
        index_dir=args.index_dir,
        dataset_path=args.dataset,
        company=args.company,
        top_k=args.top_k,
        base_url=args.base_url,
    )
    json_path, csv_path = write_evaluation_outputs(payload, args.output)

    summary = payload["summary"]
    print(
        f"Evaluated retrieval for {payload['company']} on {payload['question_count']} questions "
        f"(top_k={payload['top_k']}, embedding_model={payload['embedding_model']})."
    )
    print(
        f"Hits: {summary['hit']} | Partial: {summary['partial']} | "
        f"Misses: {summary['miss']} | Hit rate: {summary['hit_rate']:.1%}"
    )
    print(f"Saved evaluation to {json_path} and {csv_path}")

    failures = [row for row in payload["results"] if row["status"] != "hit"]
    if failures:
        print("\nQuestions needing attention:")
        for row in failures:
            print(
                f"- [{row['status']}] score={row['best_match_score']:.2f} | "
                f"{row['topic']} | {row['question']}"
            )
    return 0
