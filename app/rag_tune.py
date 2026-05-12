"""Iterative tuning workflow for the local ESG RAG pipeline."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.rag import (
    DEFAULT_BASE_URL,
    DEFAULT_SAMPLE_DIR,
    AlbertClient,
    answer_question,
    build_index,
    require_api_key,
    retrieve_chunks,
)
from app.rag_eval import (
    DEFAULT_EVAL_DATASET,
    _load_rows,
    _parse_expected_contexts,
    _score_retrieval,
    score_answer_text,
)
from app.utils import OUTPUT_DIR

DEFAULT_TUNE_OUTPUT = OUTPUT_DIR / "rag_tune_results.json"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "item"


def _pick_pdf_for_company(company: str, pdf_paths: list[Path]) -> Path:
    best_path = None
    best_score = -1.0
    for pdf_path in pdf_paths:
        stem = pdf_path.stem.replace("_", " ").replace("-", " ")
        score = max(
            float(fuzz.token_set_ratio(company.lower(), stem.lower())),
            float(fuzz.partial_ratio(company.lower(), stem.lower())),
        )
        if score > best_score:
            best_score = score
            best_path = pdf_path
    if best_path is None:
        raise RuntimeError(f"Could not match a sample PDF to company '{company}'.")
    return best_path


def _derive_chunk_bounds(target_tokens: int) -> tuple[int, int]:
    min_tokens = max(120, int(round(target_tokens * 0.72)))
    max_tokens = max(target_tokens + 40, int(round(target_tokens * 1.22)))
    if min_tokens >= target_tokens:
        min_tokens = max(100, target_tokens - 40)
    if max_tokens <= target_tokens:
        max_tokens = target_tokens + 40
    return min_tokens, max_tokens


def _build_chunk_config(target_tokens: int, overlap_tokens: int) -> dict[str, int]:
    min_tokens, max_tokens = _derive_chunk_bounds(target_tokens)
    return {
        "target_tokens": target_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
    }


def _chunk_config_label(chunk_config: dict[str, int]) -> str:
    return (
        f"target={chunk_config['target_tokens']}, min={chunk_config['min_tokens']}, "
        f"max={chunk_config['max_tokens']}, overlap={chunk_config['overlap_tokens']}"
    )


def _summarize_stage_result(result: dict[str, Any]) -> str:
    summary = result["summary"]
    parts = [
        f"retrieval={summary['avg_retrieval_score']:.2f}",
        f"retrieval_hit_rate={summary['retrieval_hit_rate']:.1%}",
    ]
    if summary["avg_answer_score"] is not None:
        parts.append(f"answer={summary['avg_answer_score']:.2f}")
        parts.append(f"answer_hit_rate={summary['answer_hit_rate']:.1%}")
    parts.append(f"composite={summary['composite_score']:.2f}")
    return " | ".join(parts)


def _refine_grid(best_value: int, *, step: int, minimum: int) -> list[int]:
    return sorted({max(minimum, best_value - step), best_value, best_value + step})


def _build_or_reuse_index(
    *,
    company: str,
    pdf_path: Path,
    chunk_config: dict[str, int],
    index_root: Path,
    built_cache: set[tuple[str, int, int]],
    embedding_model: str,
    batch_size: int,
    base_url: str,
) -> Path:
    cache_key = (company, chunk_config["target_tokens"], chunk_config["overlap_tokens"])
    index_dir = index_root / _slugify(company) / (
        f"t{chunk_config['target_tokens']}_ov{chunk_config['overlap_tokens']}"
    )
    if cache_key not in built_cache:
        build_index(
            pdf_paths=[pdf_path],
            index_dir=index_dir,
            target_tokens=chunk_config["target_tokens"],
            min_tokens=chunk_config["min_tokens"],
            max_tokens=chunk_config["max_tokens"],
            overlap_tokens=chunk_config["overlap_tokens"],
            batch_size=batch_size,
            embedding_model=embedding_model,
            dry_run=False,
            base_url=base_url,
        )
        built_cache.add(cache_key)
    return index_dir


def evaluate_configuration(
    *,
    company_rows: dict[str, list[dict[str, str]]],
    company_pdfs: dict[str, Path],
    chunk_config: dict[str, int],
    top_k: int,
    embedding_model: str,
    retrieval_architecture: str,
    search_breadth: int,
    text_model: str,
    temperature: float,
    index_root: Path,
    built_cache: set[tuple[str, int, int]],
    batch_size: int,
    base_url: str,
    generate_answers: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for company, rows in company_rows.items():
        pdf_path = company_pdfs[company]
        index_dir = _build_or_reuse_index(
            company=company,
            pdf_path=pdf_path,
            chunk_config=chunk_config,
            index_root=index_root,
            built_cache=built_cache,
            embedding_model=embedding_model,
            batch_size=batch_size,
            base_url=base_url,
        )

        for row in rows:
            expected_contexts = _parse_expected_contexts(row)
            retrieved_chunks, resolved_embedding_model = retrieve_chunks(
                index_dir=index_dir,
                question=row["question"],
                top_k=top_k,
                embedding_model=embedding_model,
                retrieval_architecture=retrieval_architecture,
                search_breadth=search_breadth,
                base_url=base_url,
            )
            retrieval_scoring = _score_retrieval(expected_contexts, retrieved_chunks)

            answer = None
            answer_scoring = None
            resolved_text_model = text_model
            if generate_answers:
                answer, resolved_text_model = answer_question(
                    question=row["question"],
                    retrieved_chunks=retrieved_chunks,
                    text_model=text_model,
                    temperature=temperature,
                    base_url=base_url,
                )
                answer_scoring = score_answer_text(answer, row.get("ground_truth", ""), expected_contexts)

            results.append(
                {
                    "company": company,
                    "question": row["question"],
                    "topic": row.get("topic", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "retrieval_status": retrieval_scoring["status"],
                    "retrieval_score": retrieval_scoring["best_match_score"],
                    "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
                    "best_chunk_id": retrieval_scoring["best_chunk_id"],
                    "top_score": round(float(retrieved_chunks[0]["score"]), 4) if retrieved_chunks else None,
                    "answer_status": answer_scoring["status"] if answer_scoring else None,
                    "answer_score": answer_scoring["score"] if answer_scoring else None,
                    "answer": answer,
                    "embedding_model": resolved_embedding_model,
                    "text_model": resolved_text_model,
                }
            )

    retrieval_scores = [float(row["retrieval_score"]) for row in results]
    answer_scores = [float(row["answer_score"]) for row in results if row["answer_score"] is not None]
    retrieval_counts = Counter(row["retrieval_status"] for row in results)
    answer_counts = Counter(row["answer_status"] for row in results if row["answer_status"])

    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    avg_answer_score = (sum(answer_scores) / len(answer_scores)) if answer_scores else None
    retrieval_hit_rate = retrieval_counts.get("hit", 0) / len(results) if results else 0.0
    answer_hit_rate = (
        answer_counts.get("hit", 0) / len(answer_scores) if answer_scores else None
    )

    if avg_answer_score is None:
        composite_score = avg_retrieval_score
    else:
        composite_score = (0.45 * avg_retrieval_score) + (0.55 * avg_answer_score)

    by_company: dict[str, dict[str, float]] = {}
    for company in company_rows:
        company_result_rows = [row for row in results if row["company"] == company]
        company_answer_scores = [float(row["answer_score"]) for row in company_result_rows if row["answer_score"] is not None]
        by_company[company] = {
            "avg_retrieval_score": round(
                sum(float(row["retrieval_score"]) for row in company_result_rows) / len(company_result_rows),
                2,
            ),
            "avg_answer_score": (
                round(sum(company_answer_scores) / len(company_answer_scores), 2) if company_answer_scores else None
            ),
        }

    return {
        "chunk_config": chunk_config,
        "retrieval_architecture": retrieval_architecture,
        "search_breadth": search_breadth,
        "temperature": temperature,
        "summary": {
            "question_count": len(results),
            "avg_retrieval_score": round(avg_retrieval_score, 2),
            "retrieval_hit_rate": round(retrieval_hit_rate, 3),
            "avg_answer_score": round(avg_answer_score, 2) if avg_answer_score is not None else None,
            "answer_hit_rate": round(answer_hit_rate, 3) if answer_hit_rate is not None else None,
            "composite_score": round(composite_score, 2),
        },
        "by_company": by_company,
        "results": results,
    }


def _pick_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("No tuning results were produced for this stage.")
    return max(results, key=lambda item: item["summary"]["composite_score"])


def _stage_candidates_from_chunk_grid(targets: list[int], overlaps: list[int]) -> list[dict[str, int]]:
    return [_build_chunk_config(target, overlap) for target in targets for overlap in overlaps]


def run_rag_tune(args: Any) -> int:
    require_api_key()
    rows = _load_rows(args.dataset)
    companies = sorted({row["company"] for row in rows if row.get("company")})
    if args.company:
        companies = [company for company in companies if company == args.company]
        if not companies:
            raise RuntimeError(f"No rows found for company '{args.company}'.")

    company_rows = {company: [row for row in rows if row.get("company") == company] for company in companies}
    pdf_paths = sorted(args.sample_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs were found under {args.sample_dir}.")
    company_pdfs = {company: _pick_pdf_for_company(company, pdf_paths) for company in companies}

    client = AlbertClient(api_key=require_api_key(), base_url=args.base_url)
    embedding_model = args.embedding_model or client.get_embedding_model(preferred="bge-m3")
    text_model = args.text_model or client.get_text_generation_model()
    index_root = args.index_root
    index_root.mkdir(parents=True, exist_ok=True)
    built_cache: set[tuple[str, int, int]] = set()

    chunk_targets = args.chunk_targets or [320, 420, 520]
    chunk_overlaps = args.chunk_overlaps or [0, 60, 120]
    retrieval_architectures = (
        getattr(args, "retrieval_modes", None)
        or getattr(args, "retrieval_architectures", None)
        or ["dense", "hybrid", "lexical"]
    )
    search_breadths = args.search_breadths or [5, 8, 12]
    temperatures = args.temperatures or [0.0, 0.2, 0.5]

    stages: list[dict[str, Any]] = []

    coarse_chunk_candidates = _stage_candidates_from_chunk_grid(chunk_targets, chunk_overlaps)
    coarse_results: list[dict[str, Any]] = []
    print(f"Stage 1/4: coarse chunking across {len(coarse_chunk_candidates)} configurations")
    for index, chunk_config in enumerate(coarse_chunk_candidates, start=1):
        result = evaluate_configuration(
            company_rows=company_rows,
            company_pdfs=company_pdfs,
            chunk_config=chunk_config,
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture="semantic",
            search_breadth=max(args.top_k, search_breadths[0]),
            text_model=text_model,
            temperature=temperatures[0],
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=False,
        )
        coarse_results.append(result)
        print(f"  [{index}/{len(coarse_chunk_candidates)}] {_chunk_config_label(chunk_config)} -> {_summarize_stage_result(result)}")
    best_chunk_result = _pick_best(coarse_results)
    stages.append({"name": "coarse_chunking", "candidates": coarse_results, "best": best_chunk_result})

    retrieval_results: list[dict[str, Any]] = []
    retrieval_candidates = [
        (architecture, breadth)
        for architecture in retrieval_architectures
        for breadth in search_breadths
        if breadth >= args.top_k
    ]
    print(f"Stage 2/4: retrieval architecture and breadth across {len(retrieval_candidates)} configurations")
    for index, (architecture, breadth) in enumerate(retrieval_candidates, start=1):
        result = evaluate_configuration(
            company_rows=company_rows,
            company_pdfs=company_pdfs,
            chunk_config=best_chunk_result["chunk_config"],
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture=architecture,
            search_breadth=breadth,
            text_model=text_model,
            temperature=temperatures[0],
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=False,
        )
        retrieval_results.append(result)
        print(f"  [{index}/{len(retrieval_candidates)}] {architecture}, breadth={breadth} -> {_summarize_stage_result(result)}")
    best_retrieval_result = _pick_best(retrieval_results)
    stages.append({"name": "retrieval_architecture", "candidates": retrieval_results, "best": best_retrieval_result})

    temperature_results: list[dict[str, Any]] = []
    print(f"Stage 3/4: answer temperature across {len(temperatures)} configurations")
    for index, temperature in enumerate(temperatures, start=1):
        result = evaluate_configuration(
            company_rows=company_rows,
            company_pdfs=company_pdfs,
            chunk_config=best_chunk_result["chunk_config"],
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture=best_retrieval_result["retrieval_architecture"],
            search_breadth=best_retrieval_result["search_breadth"],
            text_model=text_model,
            temperature=temperature,
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=True,
        )
        temperature_results.append(result)
        print(f"  [{index}/{len(temperatures)}] temperature={temperature:.2f} -> {_summarize_stage_result(result)}")
    best_temperature_result = _pick_best(temperature_results)
    stages.append({"name": "temperature", "candidates": temperature_results, "best": best_temperature_result})

    refined_targets = _refine_grid(best_chunk_result["chunk_config"]["target_tokens"], step=60, minimum=180)
    refined_overlaps = _refine_grid(best_chunk_result["chunk_config"]["overlap_tokens"], step=30, minimum=0)
    refined_chunk_candidates = _stage_candidates_from_chunk_grid(refined_targets, refined_overlaps)
    refined_results: list[dict[str, Any]] = []
    print(f"Stage 4/4: refined chunking across {len(refined_chunk_candidates)} configurations")
    for index, chunk_config in enumerate(refined_chunk_candidates, start=1):
        result = evaluate_configuration(
            company_rows=company_rows,
            company_pdfs=company_pdfs,
            chunk_config=chunk_config,
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture=best_retrieval_result["retrieval_architecture"],
            search_breadth=best_retrieval_result["search_breadth"],
            text_model=text_model,
            temperature=best_temperature_result["temperature"],
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=True,
        )
        refined_results.append(result)
        print(f"  [{index}/{len(refined_chunk_candidates)}] {_chunk_config_label(chunk_config)} -> {_summarize_stage_result(result)}")
    best_refined_result = _pick_best(refined_results)
    stages.append({"name": "refined_chunking", "candidates": refined_results, "best": best_refined_result})

    all_ranked_results = [best_temperature_result, best_refined_result]
    overall_best = _pick_best(all_ranked_results)

    payload = {
        "dataset": str(args.dataset),
        "sample_dir": str(args.sample_dir),
        "companies": companies,
        "embedding_model": embedding_model,
        "text_model": text_model,
        "top_k": args.top_k,
        "stages": stages,
        "best_configuration": overall_best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBest configuration:")
    print(f"  chunking: {_chunk_config_label(overall_best['chunk_config'])}")
    print(
        f"  retrieval: {overall_best['retrieval_architecture']} "
        f"(breadth={overall_best['search_breadth']})"
    )
    print(f"  temperature: {overall_best['temperature']:.2f}")
    print(f"  metrics: {_summarize_stage_result(overall_best)}")
    print(f"Saved tuning results to {args.output}")
    return 0
