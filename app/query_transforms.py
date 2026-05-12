"""Query transformation strategies: HyDE, Multi-Query Retrieval (MQR), and combined modes.

Uses Albert API for generation — no external paid services required.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from app.rag import AlbertClient, require_api_key
from app.utils import OUTPUT_DIR

HYDE_CACHE_DIR = OUTPUT_DIR / "hyde_cache"
DEFAULT_NUM_REFORMULATIONS = 3

_hyde_cache: dict[str, str] = {}


def _cache_key(question: str, transform: str) -> str:
    return hashlib.sha256(f"{transform}:{question}".encode()).hexdigest()[:16]


def _load_cache() -> dict[str, str]:
    cache_path = HYDE_CACHE_DIR / "cache.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    HYDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (HYDE_CACHE_DIR / "cache.json").write_text(json.dumps(cache, indent=2), encoding="utf-8")


def generate_hyde_answer(
    question: str,
    client: AlbertClient,
    model: str,
    temperature: float = 0.2,
    use_cache: bool = True,
) -> str:
    """Generate a hypothetical ideal ESG answer for HyDE retrieval."""
    global _hyde_cache
    if not _hyde_cache:
        _hyde_cache = _load_cache()

    key = _cache_key(question, "hyde")
    if use_cache and key in _hyde_cache:
        return _hyde_cache[key]

    prompt = f"""You are an ESG analyst. Write a concise hypothetical answer to this question as if you had access to a perfect sustainability report. Include specific numbers, metrics, and targets where relevant. Keep it under 150 words.

Question: {question}

Hypothetical answer:"""

    try:
        answer = client.chat_completion(model, [{"role": "user", "content": prompt}], temperature=temperature)
        cleaned = answer.strip()
        if use_cache:
            _hyde_cache[key] = cleaned
            _save_cache(_hyde_cache)
        return cleaned
    except Exception as exc:
        return f"Hypothetical ESG answer regarding: {question}"


def generate_multi_queries(
    question: str,
    client: AlbertClient,
    model: str,
    num_queries: int = DEFAULT_NUM_REFORMULATIONS,
    temperature: float = 0.3,
    use_cache: bool = True,
) -> list[dict[str, str]]:
    """Generate semantically diverse ESG-focused query reformulations."""
    global _hyde_cache
    if not _hyde_cache:
        _hyde_cache = _load_cache()

    key = _cache_key(question, f"mqr_{num_queries}")
    if use_cache and key in _hyde_cache:
        try:
            return json.loads(_hyde_cache[key])
        except (json.JSONDecodeError, Exception):
            pass

    intents = ["factual", "metric-focused", "compliance-focused", "risk-focused", "target-focused"]
    selected_intents = intents[:num_queries] if num_queries <= len(intents) else intents * (num_queries // len(intents) + 1)
    selected_intents = selected_intents[:num_queries]

    prompt = f"""Reformulate this ESG question into {num_queries} distinct retrieval queries, each with a different analytical focus:

Original question: {question}

Generate one query for each of these perspectives:
{chr(10).join(f'{i+1}. {intent}' for i, intent in enumerate(selected_intents))}

Output as JSON array:
[{{"intent": "<intent>", "query": "<reformulated query>"}}]

Each reformulated query should be specific, searchable, and capture a different dimension of the original question. Only output valid JSON."""

    try:
        raw = client.chat_completion(model, [{"role": "user", "content": prompt}], temperature=temperature)
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            if use_cache:
                _hyde_cache[key] = json.dumps(parsed)
                _save_cache(_hyde_cache)
            return parsed
    except (json.JSONDecodeError, Exception):
        pass

    fallback = [{"intent": "factual", "query": question}]
    return fallback


def combine_hyde_mqr(
    question: str,
    client: AlbertClient,
    model: str,
    num_queries: int = DEFAULT_NUM_REFORMULATIONS,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Generate both HyDE answer and MQR reformulations, combined."""
    hyde_answer = generate_hyde_answer(question, client, model, temperature=temperature)
    mqr_queries = generate_multi_queries(question, client, model, num_queries=num_queries, temperature=temperature)

    return {
        "original_question": question,
        "hyde_answer": hyde_answer,
        "hyde_query": hyde_answer,
        "mqr_queries": mqr_queries,
        "mqr_query_strings": [q["query"] for q in mqr_queries],
        "all_query_strings": [hyde_answer] + [q["query"] for q in mqr_queries],
    }


def score_query_diversity(queries: list[str]) -> float:
    if len(queries) < 2:
        return 1.0

    words_sets = [set(q.lower().split()) for q in queries]
    overlaps = []
    for i in range(len(words_sets)):
        for j in range(i + 1, len(words_sets)):
            if words_sets[i] and words_sets[j]:
                overlap = len(words_sets[i] & words_sets[j]) / len(words_sets[i] | words_sets[j])
                overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return round(1.0 - avg_overlap, 4)
