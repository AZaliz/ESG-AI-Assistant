"""Danone extractor.

Two hubs to merge:
  1. Investor financial-and-extra-financial-reports (URDs, Integrated Reports)
  2. Sustainability our-reports (thematic deep-dives)

Both static HTML. Dedupe by URL.
"""
import logging
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY, PRIORITY_ARCHIVE

logger = logging.getLogger(__name__)


TITLE_PATTERNS = [
    (r"universal registration document|urd",     "urd",                  PRIORITY_PRIMARY),
    (r"integrated annual report|annual integrated", "integrated_report", PRIORITY_PRIMARY),
    (r"climate transition plan",                  "climate_report",      PRIORITY_PRIMARY),
    (r"methane",                                  "thematic_report",     PRIORITY_SECONDARY),
    (r"palm oil|forest",                          "thematic_report",     PRIORITY_SECONDARY),
    (r"animal welfare",                           "thematic_report",     PRIORITY_SECONDARY),
    (r"human rights|salient",                     "thematic_report",     PRIORITY_SECONDARY),
    (r"bms compliance",                           "thematic_report",     PRIORITY_SECONDARY),
    (r"health journey",                           "thematic_report",     PRIORITY_SECONDARY),
    (r"extra.financial data|exhaustive",          "esg_databook",        PRIORITY_PRIMARY),
    (r"affordability|accessibility",              "thematic_report",     PRIORITY_SECONDARY),
    (r"grievance",                                "policy",              PRIORITY_ARCHIVE),
    (r"vigilance",                                "vigilance_plan",      PRIORITY_PRIMARY),
    (r"company dashboard|sustainability performance", "sustainability_report", PRIORITY_PRIMARY),
    (r"employees health and wellbeing",           "thematic_report",     PRIORITY_SECONDARY),
    (r"sustainability report",                    "sustainability_report",PRIORITY_PRIMARY),
]


class DanoneScraper(BaseScraper):
    company = "Danone"
    hub_urls = [
        "https://www.danone.com/investors/publications-and-events/financial-and-extra-financial-reports.html",
        "https://www.danone.com/sustainability/our-approach/policies-positions-reports/our-reports.html",
    ]

    def discover(self) -> list:
        docs = []
        seen = set()

        for hub in self.hub_urls:
            try:
                soup = self.fetch_html(hub)
            except Exception as e:
                logger.warning(f"  [{self.company}] hub fetch failed for {hub}: {e}")
                continue

            for a in soup.find_all("a", href=True):
                href = urljoin(hub, a["href"])
                if not re.search(r"\.(pdf|xlsx)(?:\?|#|$)", href.lower()):
                    continue
                if href in seen:
                    continue
                seen.add(href)

                title = a.get_text(strip=True)
                if not title or len(title) < 5:
                    continue

                doc_type, priority = self._classify(title)
                if doc_type is None:
                    continue

                year = self.extract_year(title, href)

                docs.append(Document(
                    company=self.company,
                    url=href,
                    title=title[:200],
                    doc_type=doc_type,
                    fiscal_year=year,
                    language="en",
                    file_format=self.file_format_from_url(href),
                    priority=priority,
                    source_hub=hub,
                ))

        return docs

    @staticmethod
    def _classify(title: str):
        t = title.lower()
        for pattern, doc_type, priority in TITLE_PATTERNS:
            if re.search(pattern, t):
                return doc_type, priority
        return None, None
