"""MCP tool implementations for ESG RAG assistant.

Each tool follows the MCP pattern: accepts structured input, validates, executes, returns structured output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.mcp.schemas import ALL_TOOLS
from app.rag import AlbertClient, DEFAULT_BASE_URL, DEFAULT_INDEX_DIR, answer_question, require_api_key, retrieve_chunks


def search_esg_reports(company: str, query: str, report_year: int | None = None, esg_pillar: str = "all", top_k: int = 5) -> dict[str, Any]:
    api_key = require_api_key()
    client = AlbertClient(api_key=api_key)

    retrieved, emb_model = retrieve_chunks(
        index_dir=DEFAULT_INDEX_DIR,
        question=f"{company}: {query}",
        top_k=top_k,
        retrieval_architecture="hybrid",
        search_breadth=20,
        base_url=DEFAULT_BASE_URL,
    )

    if report_year:
        retrieved = [c for c in retrieved if str(report_year) in c.get("source_file", "") or str(report_year) in c.get("text", "")]
    if esg_pillar != "all":
        topic_keywords = {
            "environmental": ["emission", "carbon", "climate", "ghg", "scope", "energy", "biodiversity"],
            "social": ["employee", "safety", "trir", "diversity", "inclusion", "community", "worker"],
            "governance": ["board", "committee", "risk", "compliance", "audit", "tcfd", "taxonomy"],
        }
        keywords = topic_keywords.get(esg_pillar, [])
        retrieved = [c for c in retrieved if any(kw in c["text"].lower() for kw in keywords)]

    return {
        "query": query,
        "company": company,
        "result_count": len(retrieved),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "pages": f"{c['page_start']}-{c['page_end']}",
                "score": c["score"],
                "text_preview": c["text"][:300],
            }
            for c in retrieved[:top_k]
        ],
    }


def answer_question_tool(question: str, top_k: int = 5, temperature: float = 0.3) -> dict[str, Any]:
    api_key = require_api_key()
    retrieved, emb_model = retrieve_chunks(
        index_dir=DEFAULT_INDEX_DIR,
        question=question,
        top_k=top_k,
        retrieval_architecture="dense",
        search_breadth=15,
        base_url=DEFAULT_BASE_URL,
    )
    answer, text_model = answer_question(
        question=question,
        retrieved_chunks=retrieved,
        temperature=temperature,
        base_url=DEFAULT_BASE_URL,
    )

    return {
        "question": question,
        "answer": answer,
        "embedding_model": emb_model,
        "text_model": text_model,
        "retrieved_chunks": [
            {"chunk_id": c["chunk_id"], "source": c["source_file"], "pages": f"{c['page_start']}-{c['page_end']}", "score": c["score"]}
            for c in retrieved
        ],
    }


def compare_companies(companies: list[str], metric: str) -> dict[str, Any]:
    comparisons: dict[str, dict[str, Any]] = {}
    for company in companies:
        retrieved, _ = retrieve_chunks(
            index_dir=DEFAULT_INDEX_DIR,
            question=f"{company} {metric} ESG metric",
            top_k=5,
            retrieval_architecture="dense",
            search_breadth=15,
            base_url=DEFAULT_BASE_URL,
        )
        comparisons[company] = {
            "chunks_found": len(retrieved),
            "top_scores": [c["score"] for c in retrieved[:3]],
            "relevant_texts": [c["text"][:200] for c in retrieved[:3]],
        }

    return {"metric": metric, "companies": companies, "comparisons": comparisons}


TOOL_REGISTRY = {
    "search_esg_reports": search_esg_reports,
    "retrieve_chunks": lambda **kwargs: retrieve_chunks(
        index_dir=DEFAULT_INDEX_DIR,
        question=kwargs["question"],
        top_k=kwargs.get("top_k", 5),
        retrieval_architecture=kwargs.get("retrieval_mode", "dense"),
        search_breadth=kwargs.get("candidate_k", 15),
        base_url=DEFAULT_BASE_URL,
    ),
    "answer_question": answer_question_tool,
    "compare_companies": compare_companies,
}


def list_tools() -> list[dict[str, Any]]:
    return ALL_TOOLS


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOL_REGISTRY.keys())}")

    result = tool(**arguments)
    if isinstance(result, tuple):
        result = {"chunks": list(result[0]), "embedding_model": result[1]}

    return {"tool": name, "arguments": arguments, "result": result}
