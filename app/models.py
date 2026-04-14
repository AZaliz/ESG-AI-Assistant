"""Pydantic models used across the ESG acquisition pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


DocumentType = Literal[
    "sustainability_report",
    "annual_report",
    "climate_report",
    "gri_index",
    "esg_databook",
    "unknown",
]

SourceType = Literal[
    "issuer_pdf",
    "issuer_html",
    "filing_portal",
    "search_fallback",
    "ranking_directory",
]

ParseStatus = Literal["success", "partial", "failed"]


class CompanySeed(BaseModel):
    name: str
    ticker: str | None = None
    country: str | None = None
    market_cap: str | None = None
    ranking_source: str
    ranking_url: str
    company_page_url: str | None = None
    issuer_domain: str | None = None


class CompanyProfile(BaseModel):
    seed: CompanySeed
    canonical_name: str
    issuer_domains: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CandidateLink(BaseModel):
    company_name: str
    url: str
    source_page_url: str
    source_type: SourceType
    anchor_text: str = ""
    page_title: str = ""
    nearby_text: str = ""
    file_name: str = ""
    mime_hint: str | None = None
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    report_year: int | None = None
    document_type: DocumentType = "unknown"


class ParseResult(BaseModel):
    parser_name: str
    status: ParseStatus
    text: str = ""
    notes: str = ""
    page_count: int | None = None
    extracted_title: str | None = None


class DocumentRecord(BaseModel):
    company_name: str
    ticker: str | None = None
    country: str | None = None
    report_year: int | None = None
    document_type: DocumentType = "unknown"
    title: str
    source_url: str
    final_url: str
    source_type: SourceType
    mime_type: str
    download_path: str
    text_path: str | None = None
    file_hash: str
    published_date: str | None = None
    discovery_confidence: float = 0.0
    parse_status: ParseStatus
    notes: str = ""
    parser_used: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SmokeTestRow(BaseModel):
    company: str
    discovered_issuer_domain: str | None = None
    chosen_report_title: str | None = None
    report_year: int | None = None
    source_url: str | None = None
    file_type: str | None = None
    confidence: float | None = None
    parse_status: ParseStatus | None = None
    notes: str = ""
