"""Agentic retrieval loop: ReAct-style ESG retrieval with reflection, query repair, and evidence sufficiency scoring.

Features:
- Multi-turn retrieval with reasoning
- Evidence sufficiency scoring
- Adaptive retrieval breadth
- Query repair on weak results
- Retrieval reflection and retry
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from app.rag import AlbertClient, DEFAULT_BASE_URL, answer_question, require_api_key, retrieve_chunks


def score_evidence_sufficiency(
    question: str, retrieved_chunks: list[dict[str, Any]], client: AlbertClient, model: str
) -> dict[str, Any]:
    if not retrieved_chunks:
        return {"sufficient": False, "score": 0.0, "reasoning": "No chunks retrieved."}

    chunk_summaries = "\n\n".join(
        f"[{c['chunk_id']}] score={c['score']:.4f} | {c['text'][:200]}..."
        for c in retrieved_chunks[:5]
    )

    prompt = f"""Evaluate whether the retrieved ESG chunks provide sufficient evidence to answer this question.

Question: {question}

Retrieved chunks:
{chunk_summaries}

Output as JSON:
{{"sufficient": true|false, "score": <float 0-1>, "missing_aspects": ["<what's missing>"], "recommendation": "<what to search for next>"}}

Only output valid JSON. No markdown."""

    try:
        raw = client.chat_completion(model, [{"role": "user", "content": prompt}], temperature=0.0)
        return json.loads(raw)
    except (json.JSONDecodeError, Exception):
        top_score = max(c["score"] for c in retrieved_chunks) if retrieved_chunks else 0.0
        return {
            "sufficient": top_score > 0.5,
            "score": top_score,
            "missing_aspects": [],
            "recommendation": "",
        }


def repair_query(original_question: str, reflection: dict[str, Any], client: AlbertClient, model: str) -> str:
    missing = ", ".join(reflection.get("missing_aspects", ["specific metrics and targets"]))
    recommendation = reflection.get("recommendation", "")

    prompt = f"""Repair this ESG question to improve retrieval. The original question had weak retrieval results.

Original question: {original_question}

Missing information: {missing}
Search recommendation: {recommendation}

Generate a revised search query that would better retrieve the needed ESG information. Make it more specific and keyword-focused.

Output only the revised query text, no JSON, no explanation."""

    try:
        return client.chat_completion(model, [{"role": "user", "content": prompt}], temperature=0.2).strip()
    except Exception:
        return f"{original_question} {recommendation}"


def run_agentic_retrieval(
    *,
    index_dir: str,
    question: str,
    top_k: int = 5,
    candidate_k: int = 20,
    retrieval_mode: str = "dense",
    max_iterations: int = 3,
    evidence_threshold: float = 0.5,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    text_model = client.get_text_generation_model()

    trace: list[dict[str, Any]] = []
    current_question = question
    all_chunks: dict[str, dict[str, Any]] = {}
    final_answer = ""
    sufficient = False

    for iteration in range(1, max_iterations + 1):
        step: dict[str, Any] = {
            "iteration": iteration,
            "query": current_question,
            "action": "retrieve",
        }

        retrieved, _ = retrieve_chunks(
            index_dir=index_dir,
            question=current_question,
            top_k=top_k,
            retrieval_architecture=retrieval_mode,
            search_breadth=candidate_k,
            base_url=base_url,
        )

        step["chunks_retrieved"] = len(retrieved)
        step["top_score"] = float(retrieved[0]["score"]) if retrieved else 0.0

        for chunk in retrieved:
            if chunk["chunk_id"] not in all_chunks:
                all_chunks[chunk["chunk_id"]] = chunk

        assessment = score_evidence_sufficiency(question, retrieved, client, text_model)
        step["assessment"] = assessment
        step["sufficient"] = assessment.get("sufficient", False)
        step["evidence_score"] = assessment.get("score", 0.0)

        if assessment.get("sufficient", False) or step["evidence_score"] >= evidence_threshold:
            sufficient = True
            step["action"] = "answer"
            trace.append(step)
            break

        if iteration < max_iterations:
            step["action"] = "repair"
            current_question = repair_query(question, assessment, client, text_model)
            step["repaired_query"] = current_question
            candidate_k = min(candidate_k + 10, 50)

        trace.append(step)

    sorted_chunks = sorted(all_chunks.values(), key=lambda c: c["score"], reverse=True)
    deduplicated = []
    seen_texts: set[str] = set()
    for chunk in sorted_chunks:
        norm = " ".join(chunk["text"].split())[:200]
        if norm not in seen_texts:
            seen_texts.add(norm)
            deduplicated.append(chunk)

    if deduplicated:
        final_answer, _ = answer_question(
            question=question,
            retrieved_chunks=deduplicated[:top_k],
            text_model=text_model,
            base_url=base_url,
        )

    return {
        "question": question,
        "iterations": len(trace),
        "sufficient": sufficient,
        "final_answer": final_answer,
        "total_unique_chunks": len(all_chunks),
        "final_chunks": deduplicated[:top_k],
        "trace": trace,
    }
