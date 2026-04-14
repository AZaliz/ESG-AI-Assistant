"""Deterministic scoring for report candidates."""

from __future__ import annotations

from datetime import UTC, datetime
import re

from rapidfuzz import fuzz

from app.models import CandidateLink, CompanyProfile
from app.utils import closest_year, extract_years, normalize_whitespace


POSITIVE_SIGNALS = {
    "sustainability report": 35,
    "annual report": 18,
    "climate report": 28,
    "esg report": 30,
    "integrated report": 18,
    "gri index": 16,
    "databook": 12,
    "sustainability statement": 22,
    "non-financial statement": 10,
    "tcfd": 12,
}

NEGATIVE_SIGNALS = {
    "press release": -30,
    "news": -20,
    "blog": -20,
    "career": -25,
    "event": -15,
    "webcast": -15,
    "presentation": -8,
    "earnings call": -20,
}

PATH_SIGNALS = {
    "/sustainability/": 10,
    "/esg/": 10,
    "/investor": 6,
    "/reports/": 8,
    "/publications/": 6,
}


def rank_candidates(candidates: list[CandidateLink], profile: CompanyProfile) -> list[CandidateLink]:
    scored = [score_candidate(candidate, profile) for candidate in candidates]
    return sorted(scored, key=lambda item: item.score, reverse=True)


def score_candidate(candidate: CandidateLink, profile: CompanyProfile) -> CandidateLink:
    raw_haystack = normalize_whitespace(
        " ".join(
            [
                candidate.url,
                candidate.anchor_text,
                candidate.page_title,
                candidate.nearby_text,
                candidate.file_name,
            ]
        )
    ).lower()
    haystack = re.sub(r"[-_/.:]+", " ", raw_haystack)
    reasons: list[str] = []
    score = 0.0

    for phrase, weight in POSITIVE_SIGNALS.items():
        if phrase in haystack:
            score += weight
            reasons.append(f"+{weight} keyword:{phrase}")

    for phrase, weight in NEGATIVE_SIGNALS.items():
        if phrase in haystack:
            score += weight
            reasons.append(f"{weight} negative:{phrase}")

    for path_hint, weight in PATH_SIGNALS.items():
        if path_hint in candidate.url.lower():
            score += weight
            reasons.append(f"+{weight} path:{path_hint}")

    if candidate.url.lower().endswith(".pdf") or candidate.mime_hint == "application/pdf":
        score += 18
        reasons.append("+18 pdf")
    elif candidate.url.lower().endswith((".html", ".htm", "/")):
        score += 4
        reasons.append("+4 html-like")

    company_similarity = fuzz.partial_ratio(profile.canonical_name.lower(), haystack)
    if company_similarity >= 85:
        score += 10
        reasons.append(f"+10 company-match:{company_similarity}")
    elif company_similarity >= 70:
        score += 5
        reasons.append(f"+5 company-match:{company_similarity}")

    current_year = datetime.now(UTC).year
    years = extract_years(candidate.url, candidate.anchor_text, candidate.page_title, candidate.nearby_text)
    year = closest_year(years, current_year)
    candidate.report_year = year
    if year:
        recency = max(0, 12 - abs(current_year - year) * 3)
        score += recency
        reasons.append(f"+{recency} year:{year}")
    else:
        score -= 5
        reasons.append("-5 missing-year")

    candidate.document_type = infer_document_type(haystack)
    if candidate.document_type != "unknown":
        score += 6
        reasons.append(f"+6 type:{candidate.document_type}")

    if score < 0:
        score = 0.0

    candidate.score = round(score, 2)
    candidate.reasons = reasons
    return candidate


def infer_document_type(haystack: str) -> str:
    checks = [
        (
            "sustainability_report",
            ("sustainability report", "sustainability statement", "csr report", "esg report", "esg reports", "esg"),
        ),
        ("annual_report", ("annual report", "integrated report", "universal registration document")),
        ("climate_report", ("climate report", "climate transition", "tcfd", "climate-related")),
        ("gri_index", ("gri index", "gri content index")),
        ("esg_databook", ("esg databook", "esg data book", "esg data", "data book")),
    ]
    for document_type, phrases in checks:
        if any(phrase in haystack for phrase in phrases):
            return document_type
    return "unknown"
