"""MCP-compatible tool schemas for ESG RAG assistant.

Defines JSON schemas for all exposed tools following MCP conventions.
"""

from __future__ import annotations

from typing import Any

SEARCH_ESG_SCHEMA: dict[str, Any] = {
    "name": "search_esg_reports",
    "description": "Search indexed ESG reports by company name and keywords",
    "inputSchema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Company name"},
            "query": {"type": "string", "description": "Search query or keywords"},
            "report_year": {"type": "integer", "description": "Filter by report year"},
            "esg_pillar": {"type": "string", "enum": ["environmental", "social", "governance", "all"], "default": "all"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["company", "query"],
    },
}

RETRIEVE_CHUNKS_SCHEMA: dict[str, Any] = {
    "name": "retrieve_chunks",
    "description": "Retrieve relevant ESG report chunks using dense/lexical/hybrid retrieval",
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "ESG question to answer"},
            "top_k": {"type": "integer", "default": 5},
            "retrieval_mode": {"type": "string", "enum": ["dense", "lexical", "hybrid"], "default": "dense"},
            "filter_year": {"type": "integer", "description": "Filter chunks by report year"},
            "filter_pillar": {"type": "string", "enum": ["environmental", "social", "governance", "all"], "default": "all"},
        },
        "required": ["question"],
    },
}

ANSWER_QUESTION_SCHEMA: dict[str, Any] = {
    "name": "answer_question",
    "description": "Generate grounded ESG answer from retrieved chunks with citations",
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "ESG question"},
            "top_k": {"type": "integer", "default": 5},
            "temperature": {"type": "number", "default": 0.3},
        },
        "required": ["question"],
    },
}

RAGAS_EVAL_SCHEMA: dict[str, Any] = {
    "name": "run_ragas_eval",
    "description": "Run RAGAS evaluation metrics on the ESG QA dataset",
    "inputSchema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Company to evaluate"},
            "eval_mode": {"type": "string", "enum": ["retrieval", "answer", "ragas", "all"], "default": "all"},
        },
        "required": ["company"],
    },
}

DISCOVER_COMPANY_SCHEMA: dict[str, Any] = {
    "name": "discover_company_reports",
    "description": "Discover available ESG/sustainability reports for a company",
    "inputSchema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Company name"},
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
            "country": {"type": "string", "description": "Country of incorporation"},
        },
        "required": ["company"],
    },
}

INGEST_REPORT_SCHEMA: dict[str, Any] = {
    "name": "ingest_report",
    "description": "Download, parse, and index an ESG/sustainability report",
    "inputSchema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Company name"},
            "url": {"type": "string", "description": "Direct URL to PDF report"},
            "chunk_size": {"type": "integer", "default": 420},
        },
        "required": ["company", "url"],
    },
}

COMPARE_COMPANIES_SCHEMA: dict[str, Any] = {
    "name": "compare_companies",
    "description": "Compare ESG metrics across multiple indexed companies",
    "inputSchema": {
        "type": "object",
        "properties": {
            "companies": {"type": "array", "items": {"type": "string"}, "description": "Company names to compare"},
            "metric": {"type": "string", "description": "ESG metric to compare (e.g., scope1_emissions, trir, taxonomy_alignment)"},
        },
        "required": ["companies", "metric"],
    },
}

EXTRACT_ESG_METRICS_SCHEMA: dict[str, Any] = {
    "name": "extract_esg_metrics",
    "description": "Extract structured ESG metrics from a company's indexed reports",
    "inputSchema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Company name"},
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metrics to extract (scope1, scope2, scope3, trir, taxonomy_capex, etc.)",
            },
            "year": {"type": "integer", "description": "Report year"},
        },
        "required": ["company"],
    },
}

ALL_TOOLS = [
    SEARCH_ESG_SCHEMA,
    RETRIEVE_CHUNKS_SCHEMA,
    ANSWER_QUESTION_SCHEMA,
    RAGAS_EVAL_SCHEMA,
    DISCOVER_COMPANY_SCHEMA,
    INGEST_REPORT_SCHEMA,
    COMPARE_COMPANIES_SCHEMA,
    EXTRACT_ESG_METRICS_SCHEMA,
]
