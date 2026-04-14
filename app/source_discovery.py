"""Discover sustainability-report candidates across structured, issuer, and search sources."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models import CandidateLink, CompanyProfile
from app.utils import filename_from_url, normalize_url, normalize_whitespace, same_domain


LOGGER = logging.getLogger(__name__)

DISCOVERY_PATH_HINTS = (
    "sustainability",
    "esg",
    "investor",
    "report",
    "publications",
    "annual",
    "climate",
    "responsibility",
)


class SourceDiscovery:
    def __init__(self, session: requests.Session) -> None:
        self.session = session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _fetch(self, url: str) -> requests.Response:
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response

    def discover(self, profile: CompanyProfile) -> list[CandidateLink]:
        candidates: list[CandidateLink] = []
        visited: set[str] = set()
        hub_pages: list[tuple[str, str]] = []

        for url in profile.source_urls:
            try:
                response = self._fetch(url)
            except requests.RequestException as exc:
                LOGGER.info("Failed to fetch discovery page %s: %s", url, exc)
                continue

            final_url = normalize_url(response.url)
            visited.add(final_url)

            if _is_pdf_response(response):
                candidates.append(
                    CandidateLink(
                        company_name=profile.canonical_name,
                        url=final_url,
                        source_page_url=url,
                        source_type="issuer_pdf" if any(same_domain(final_url, d) for d in profile.issuer_domains) else "ranking_directory",
                        anchor_text=filename_from_url(final_url),
                        page_title="",
                        nearby_text="Direct PDF discovered from source URL",
                        file_name=filename_from_url(final_url),
                        mime_hint="application/pdf",
                    )
                )
                continue

            page_candidates, page_hubs = self._extract_from_html(
                html=response.text,
                page_url=final_url,
                profile=profile,
                source_type=self._classify_source(url, profile),
            )
            candidates.extend(page_candidates)
            hub_pages.extend(page_hubs)

        for hub_url, source_type in hub_pages[:25]:
            if hub_url in visited:
                continue
            visited.add(hub_url)
            try:
                response = self._fetch(hub_url)
            except requests.RequestException as exc:
                LOGGER.info("Failed to fetch hub page %s: %s", hub_url, exc)
                continue

            if _is_pdf_response(response):
                continue
            page_candidates, _ = self._extract_from_html(
                html=response.text,
                page_url=normalize_url(response.url),
                profile=profile,
                source_type=source_type,
            )
            candidates.extend(page_candidates)

        if not candidates:
            candidates.extend(self._search_fallback(profile))

        unique: dict[str, CandidateLink] = {}
        for candidate in candidates:
            unique.setdefault(candidate.url, candidate)
        return list(unique.values())

    def _extract_from_html(
        self,
        html: str,
        page_url: str,
        profile: CompanyProfile,
        source_type: str,
    ) -> tuple[list[CandidateLink], list[tuple[str, str]]]:
        soup = BeautifulSoup(html, "lxml")
        page_title = normalize_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
        candidates: list[CandidateLink] = []
        hub_pages: list[tuple[str, str]] = []

        for link in soup.find_all("a", href=True):
            href = normalize_url(link["href"], page_url)
            anchor_text = normalize_whitespace(link.get_text(" ", strip=True))
            nearby_text = normalize_whitespace(link.parent.get_text(" ", strip=True)[:400]) if link.parent else ""
            href_lower = href.lower()

            if any(term in href_lower for term in ("javascript:", "mailto:", "tel:")):
                continue
            if any(term in href_lower for term in ("press-release", "/news", "/media", "/careers", "/events")):
                continue

            if source_type == "ranking_directory" and "companiesmarketcap.com" in href_lower:
                if href_lower.endswith(".pdf"):
                    file_name = filename_from_url(href)
                    candidates.append(
                        CandidateLink(
                            company_name=profile.canonical_name,
                            url=href,
                            source_page_url=page_url,
                            source_type=source_type,
                            anchor_text=anchor_text,
                            page_title=page_title,
                            nearby_text=nearby_text,
                            file_name=file_name,
                            mime_hint="application/pdf",
                        )
                    )
                elif any(token in href_lower for token in ("/esg-reports/", "/annual-reports/", "/sec-reports-20f/")):
                    hub_pages.append((href, source_type))
                continue

            file_name = filename_from_url(href)
            mime_hint = "application/pdf" if href_lower.endswith(".pdf") else None
            if _looks_like_document_link(href, anchor_text, nearby_text):
                candidates.append(
                    CandidateLink(
                        company_name=profile.canonical_name,
                        url=href,
                        source_page_url=page_url,
                        source_type=source_type,
                        anchor_text=anchor_text,
                        page_title=page_title,
                        nearby_text=nearby_text,
                        file_name=file_name,
                        mime_hint=mime_hint,
                    )
                )
            elif _looks_like_hub_link(href, anchor_text) and self._should_follow_hub(href, profile):
                hub_pages.append((href, source_type))

        return candidates, hub_pages

    def _classify_source(self, url: str, profile: CompanyProfile) -> str:
        if "companiesmarketcap.com" in url:
            return "ranking_directory"
        if any(same_domain(url, domain) for domain in profile.issuer_domains):
            if url.lower().endswith(".pdf"):
                return "issuer_pdf"
            return "issuer_html"
        return "search_fallback"

    @staticmethod
    def _should_follow_hub(url: str, profile: CompanyProfile) -> bool:
        return any(same_domain(url, domain) for domain in profile.issuer_domains) or "companiesmarketcap.com" in url

    def _search_fallback(self, profile: CompanyProfile) -> list[CandidateLink]:
        queries: list[str] = []
        for domain in profile.issuer_domains[:1]:
            queries.extend(
                [
                    f"site:{domain} sustainability report pdf",
                    f"site:{domain} esg report pdf",
                    f"site:{domain} annual report sustainability",
                ]
            )
        queries.append(f'"{profile.canonical_name}" sustainability report pdf')
        candidates: list[CandidateLink] = []

        for query in queries[:4]:
            search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
            try:
                response = self._fetch(search_url)
            except requests.RequestException as exc:
                LOGGER.info("Search fallback failed for %s: %s", query, exc)
                continue
            soup = BeautifulSoup(response.text, "lxml")
            for result in soup.select("a.result__a"):
                href = normalize_url(result.get("href", ""), response.url)
                text = normalize_whitespace(result.get_text(" ", strip=True))
                if not href:
                    continue
                candidates.append(
                    CandidateLink(
                        company_name=profile.canonical_name,
                        url=href,
                        source_page_url=search_url,
                        source_type="search_fallback",
                        anchor_text=text,
                        page_title="",
                        nearby_text=query,
                        file_name=filename_from_url(href),
                        mime_hint="application/pdf" if href.lower().endswith(".pdf") else None,
                    )
                )
        return candidates


def _is_pdf_response(response: requests.Response) -> bool:
    return response.headers.get("Content-Type", "").lower().startswith("application/pdf")


def _looks_like_document_link(href: str, anchor_text: str, nearby_text: str) -> bool:
    haystack = " ".join((href, anchor_text, nearby_text)).lower()
    negative_terms = (
        "stock-price-history",
        "pe-ratio",
        "ps-ratio",
        "pb-ratio",
        "operating-margin",
        "eps",
        "dividends",
        "shares-outstanding",
    )
    if any(term in haystack for term in negative_terms):
        return False
    if href.lower().endswith((".pdf", ".html", ".htm")):
        return True
    positive_terms = (
        "sustainability",
        "esg",
        "annual report",
        "climate",
        "gri",
        "integrated report",
        "statement",
        "report",
        "databook",
    )
    return any(term in haystack for term in positive_terms)


def _looks_like_hub_link(href: str, anchor_text: str) -> bool:
    haystack = " ".join((href, anchor_text)).lower()
    return any(hint in haystack for hint in DISCOVERY_PATH_HINTS)
