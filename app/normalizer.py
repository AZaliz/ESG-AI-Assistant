"""Normalize candidate and parser output into a document record."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.candidate_ranker import infer_document_type
from app.models import CandidateLink, CompanyProfile, DocumentRecord, ParseResult
from app.utils import closest_year, extract_years, filename_from_url, normalize_whitespace


def build_document_record(
    profile: CompanyProfile,
    candidate: CandidateLink,
    final_url: str,
    mime_type: str,
    download_path: Path,
    text_path: Path | None,
    file_hash: str,
    parse_result: ParseResult,
) -> DocumentRecord:
    title = normalize_whitespace(
        parse_result.extracted_title or candidate.anchor_text or candidate.page_title or filename_from_url(final_url)
    )
    years = extract_years(title, final_url, candidate.nearby_text)
    report_year = candidate.report_year or closest_year(years, datetime.now(UTC).year)
    published_date = f"{report_year}-01-01" if report_year else None
    document_type = candidate.document_type
    if document_type == "unknown":
        document_type = infer_document_type(title.lower())

    return DocumentRecord(
        company_name=profile.canonical_name,
        ticker=profile.seed.ticker,
        country=profile.seed.country,
        report_year=report_year,
        document_type=document_type,
        title=title,
        source_url=candidate.source_page_url,
        final_url=final_url,
        source_type=candidate.source_type,
        mime_type=mime_type,
        download_path=str(download_path),
        text_path=str(text_path) if text_path else None,
        file_hash=file_hash,
        published_date=published_date,
        discovery_confidence=candidate.score,
        parse_status=parse_result.status,
        notes=parse_result.notes,
        parser_used=parse_result.parser_name,
    )
