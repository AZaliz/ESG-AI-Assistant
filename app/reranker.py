"""Multi-stage reranking for ESG retrieval: lexical, embedding, LLM, and cross-encoder modes.

Implements true 2-stage retrieval:
1. Broad retrieval (candidate-k chunks)
2. Reranking to final top-k
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from app.rag import AlbertClient, ChunkRecord, _normalize_match_scores, _score_lexical_matches, embed_texts


def rerank_lexical(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    chunks = [
        ChunkRecord(
            chunk_id=c["chunk_id"],
            source_file=c["source_file"],
            source_path=c["source_path"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            token_count=c["token_count"],
            text=c["text"],
        )
        for c in candidates
    ]
    scores = _score_lexical_matches(question, chunks)
    normalized = _normalize_match_scores([(idx, scores[idx]) for idx in scores])
    ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

    result: list[dict[str, Any]] = []
    for idx, score in ranked[:top_k]:
        chunk = candidates[idx].copy()
        chunk["original_score"] = chunk.get("score", 0.0)
        chunk["score"] = score
        chunk["reranker"] = "lexical"
        result.append(chunk)
    return result


def rerank_embedding(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    client: AlbertClient,
    embedding_model: str,
) -> list[dict[str, Any]]:
    query_vector = embed_texts(client, [question], embedding_model=embedding_model, batch_size=1)[0]
    candidate_texts = [c["text"] for c in candidates]
    candidate_vectors = embed_texts(client, candidate_texts, embedding_model=embedding_model, batch_size=16)

    norm_query = query_vector / (np.linalg.norm(query_vector) or 1.0)
    norm_chunks = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) or 1.0)
    similarities = norm_chunks @ norm_query

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    result: list[dict[str, Any]] = []
    for idx in ranked_indices:
        chunk = candidates[int(idx)].copy()
        chunk["original_score"] = chunk.get("score", 0.0)
        chunk["score"] = float(similarities[int(idx)])
        chunk["reranker"] = "embedding"
        result.append(chunk)
    return result


def rerank_llm(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    client: AlbertClient,
    model: str,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    chunk_list = "\n\n---\n\n".join(
        f"[{i}] (pages {c['page_start']}-{c['page_end']}, score={c.get('score', 0):.4f}): {c['text'][:300]}..."
        for i, c in enumerate(candidates[:min(len(candidates), 20)])
    )

    prompt = f"""You are an ESG retrieval judge. Rank these retrieved chunks by relevance to the question.

Question: {question}

Candidate chunks:
{chunk_list}

Rank the chunks by:
1. Direct relevance to the question
2. Presence of specific ESG metrics, targets, or data
3. Factual grounding (prefer concrete data over vague narrative)
4. Alignment with ESG disclosure frameworks

Return the top {top_k} chunk indices in order of relevance, with scores 0-100:

Output as JSON:
{{"rankings": [{{"index": <int>, "score": <float 0-100>, "reason": "<brief>"}}]}}

Only output valid JSON. No markdown."""

    try:
        raw = client.chat_completion(model, [{"role": "user", "content": prompt}], temperature=temperature)
        parsed = json.loads(raw)
        rankings = parsed.get("rankings", [])

        result: list[dict[str, Any]] = []
        for rank in rankings:
            idx = int(rank["index"])
            if 0 <= idx < len(candidates):
                chunk = candidates[idx].copy()
                chunk["original_score"] = chunk.get("score", 0.0)
                chunk["score"] = float(rank["score"]) / 100.0
                chunk["reranker"] = "llm"
                chunk["reranker_reason"] = rank.get("reason", "")
                result.append(chunk)
                if len(result) >= top_k:
                    break

        remaining = [c for c in candidates if c not in result]
        for chunk in remaining[: top_k - len(result)]:
            chunk = chunk.copy()
            chunk["original_score"] = chunk.get("score", 0.0)
            chunk["reranker"] = "llm_fallback"
            result.append(chunk)

        return result[:top_k]

    except (json.JSONDecodeError, Exception):
        return candidates[:top_k]


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    reranker: str = "lexical",
    client: AlbertClient | None = None,
    embedding_model: str | None = None,
    text_model: str | None = None,
) -> list[dict[str, Any]]:
    reranker = reranker.strip().lower()

    if reranker == "none" or len(candidates) <= top_k:
        return candidates[:top_k]

    if reranker == "lexical":
        return rerank_lexical(question, candidates, top_k)

    if reranker == "embedding" and client is not None and embedding_model:
        return rerank_embedding(question, candidates, top_k, client, embedding_model)

    if reranker in ("llm", "cross_encoder") and client is not None and text_model:
        return rerank_llm(question, candidates, top_k, client, text_model)

    return candidates[:top_k]
