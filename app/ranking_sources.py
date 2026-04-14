"""Ranking-source adapters for dynamic smoke tests."""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models import CompanySeed
from app.utils import EUROPEAN_COUNTRIES, normalize_url, normalize_whitespace


LOGGER = logging.getLogger(__name__)


class CompaniesMarketCapAdapter:
    BASE_URL = "https://companiesmarketcap.com"

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

    def get_top_european_companies(self, limit: int = 10) -> list[CompanySeed]:
        companies: list[CompanySeed] = []
        page = 1
        while len(companies) < limit and page <= 8:
            url = f"{self.BASE_URL}/page/{page}/" if page > 1 else f"{self.BASE_URL}/"
            response = self._fetch(url)
            page_companies = self._parse_ranking_page(response.text, response.url)
            for company in page_companies:
                if (company.country or "") not in EUROPEAN_COUNTRIES:
                    continue
                companies.append(company)
                if len(companies) >= limit:
                    break
            page += 1
        return companies[:limit]

    def _parse_ranking_page(self, html: str, page_url: str) -> list[CompanySeed]:
        soup = BeautifulSoup(html, "lxml")
        companies: list[CompanySeed] = []
        rows = soup.find_all("tr")

        for row in rows:
            link = row.find("a", href=True)
            if not link:
                continue
            href = normalize_url(link.get("href", ""), page_url)
            if "/marketcap/" not in href:
                continue
            text = normalize_whitespace(link.get_text(" ", strip=True))
            if not text or "logo" in text.lower():
                continue

            ticker = None
            name = text
            parts = text.rsplit(" ", 1)
            if len(parts) == 2 and re.fullmatch(r"[A-Z0-9.\-]{1,12}", parts[1]):
                name, ticker = parts

            row_text = normalize_whitespace(row.get_text(" ", strip=True))
            country = self._extract_country(row_text)
            market_cap = self._extract_market_cap(row_text)
            if not country:
                continue

            companies.append(
                CompanySeed(
                    name=name,
                    ticker=ticker,
                    country=country,
                    market_cap=market_cap,
                    ranking_source="companiesmarketcap-global",
                    ranking_url=page_url,
                    company_page_url=href,
                )
            )

        return companies

    @staticmethod
    def _extract_country(row_text: str) -> str | None:
        for country in sorted(EUROPEAN_COUNTRIES, key=len, reverse=True):
            if row_text.endswith(country) or f" {country} " in f" {row_text} ":
                return country
        return None

    @staticmethod
    def _extract_market_cap(row_text: str) -> str | None:
        match = re.search(r"([$€£]\s?[0-9.,]+\s?[TMB])", row_text)
        return match.group(1).replace(" ", "") if match else None
