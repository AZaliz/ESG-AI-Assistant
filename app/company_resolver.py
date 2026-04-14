"""Resolve a company seed to canonical issuer domains and likely discovery URLs."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models import CompanyProfile, CompanySeed
from app.utils import normalize_url, normalize_whitespace, same_domain


LOGGER = logging.getLogger(__name__)
SOCIAL_HOST_HINTS = ("linkedin.com", "facebook.com", "x.com", "twitter.com", "youtube.com", "instagram.com")


class CompanyResolver:
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

    def resolve(self, seed: CompanySeed) -> CompanyProfile:
        notes: list[str] = []
        issuer_domains: list[str] = []
        source_urls: list[str] = []

        if seed.issuer_domain:
            issuer_domains.append(seed.issuer_domain)

        if seed.company_page_url:
            source_urls.append(seed.company_page_url)
            source_urls.extend(self._ranking_directory_urls(seed.company_page_url))
            try:
                response = self._fetch(seed.company_page_url)
                page_domains = self._extract_domains_from_company_page(
                    html=response.text,
                    company_name=seed.name,
                    page_url=response.url,
                )
                issuer_domains.extend(page_domains)
            except requests.RequestException as exc:
                notes.append(f"Failed to resolve canonical domain from ranking page: {exc}")

        if not issuer_domains:
            fallback_domain = self._search_official_domain(seed.name)
            if fallback_domain:
                issuer_domains.append(fallback_domain)
            else:
                notes.append("Could not confidently resolve an issuer domain; continuing with structured sources only.")

        issuer_domains = _dedupe(issuer_domains)
        if not issuer_domains and seed.company_page_url:
            parsed = urlparse(seed.company_page_url)
            if parsed.netloc and "companiesmarketcap.com" not in parsed.netloc:
                notes.append("Falling back to ranking-page domain because no issuer site was detected.")
                issuer_domains.append(parsed.netloc)

        for domain in issuer_domains:
            source_urls.extend(self._candidate_source_urls(domain))

        return CompanyProfile(
            seed=seed,
            canonical_name=normalize_whitespace(seed.name),
            issuer_domains=_dedupe(issuer_domains),
            source_urls=_dedupe(source_urls),
            notes=notes,
        )

    def _extract_domains_from_company_page(self, html: str, company_name: str, page_url: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        domains: list[str] = []

        for link in soup.find_all("a", href=True):
            href = normalize_url(link["href"], page_url)
            host = urlparse(href).netloc.lower()
            if not host or same_domain(href, urlparse(page_url).netloc):
                continue
            if any(social in host for social in SOCIAL_HOST_HINTS):
                continue
            text = normalize_whitespace(link.get_text(" ", strip=True))
            if self._looks_like_issuer_website(text=text, href=href, company_name=company_name):
                domains.append(host.replace("www.", ""))

        body_text = normalize_whitespace(soup.get_text(" ", strip=True))
        website_matches = re.findall(r"https?://[^\s\"'<>]+", body_text)
        for match in website_matches:
            host = urlparse(match).netloc.lower().replace("www.", "")
            if host and not any(social in host for social in SOCIAL_HOST_HINTS):
                domains.append(host)

        return _dedupe(domains)

    @staticmethod
    def _looks_like_issuer_website(text: str, href: str, company_name: str) -> bool:
        text_lower = text.lower()
        href_lower = href.lower()
        company_tokens = [token for token in re.split(r"[^a-z0-9]+", company_name.lower()) if len(token) > 2]
        if "website" in text_lower or "official" in text_lower:
            return True
        return any(token in href_lower for token in company_tokens)

    @staticmethod
    def _candidate_source_urls(domain: str) -> list[str]:
        base = f"https://{domain}"
        return [
            base,
            f"{base}/sustainability",
            f"{base}/sustainability/reports",
            f"{base}/esg",
            f"{base}/investors",
            f"{base}/investors/reports",
            f"{base}/investors/annual-report",
            f"{base}/publications",
            f"{base}/reports",
        ]

    @staticmethod
    def _ranking_directory_urls(company_page_url: str) -> list[str]:
        if "/marketcap/" not in company_page_url:
            return []
        base = company_page_url.rsplit("/marketcap/", 1)[0]
        return [f"{base}/esg-reports/", f"{base}/annual-reports/"]

    def _search_official_domain(self, company_name: str) -> str | None:
        query = f'"{company_name}" official site'
        search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
        try:
            response = self._fetch(search_url)
        except requests.RequestException:
            return None

        soup = BeautifulSoup(response.text, "lxml")
        company_tokens = [token for token in re.split(r"[^a-z0-9]+", company_name.lower()) if len(token) > 2]
        for link in soup.select("a.result__a"):
            href = normalize_url(link.get("href", ""), response.url)
            host = urlparse(href).netloc.lower().replace("www.", "")
            if not host or any(social in host for social in SOCIAL_HOST_HINTS):
                continue
            if any(token in host for token in company_tokens):
                return host
        return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
