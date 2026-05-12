"""Iberdrola extractor.

Annual reports hub lists per-year docs:
  - Statement of Non-financial Information / Sustainability Report
  - Integrated Report
  - Annual Corporate Governance Report
  - Annual report on remuneration of directors
  - Assurance Report
"""
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY


TITLE_PATTERNS = [
    (r"sustainability report|non.financial information","sustainability_report",PRIORITY_PRIMARY),
    (r"integrated report",                              "integrated_report",    PRIORITY_PRIMARY),
    (r"climate.*action plan|climate report",            "climate_report",       PRIORITY_PRIMARY),
    (r"annual corporate governance",                    "policy",               PRIORITY_SECONDARY),
    (r"remuneration of directors",                      "policy",               PRIORITY_SECONDARY),
    (r"assurance report",                               "framework_disclosure", PRIORITY_SECONDARY),
    (r"environmental footprint|biodiversity",           "thematic_report",      PRIORITY_SECONDARY),
    (r"ghg|greenhouse gas",                             "esg_databook",         PRIORITY_SECONDARY),
    (r"compliance",                                     "policy",               PRIORITY_SECONDARY),
    (r"community contributions",                        "thematic_report",      PRIORITY_SECONDARY),
]


class IberdrolaScraper(BaseScraper):
    company = "Iberdrola"
    hub_urls = [
        "https://www.iberdrola.com/shareholders-investors/operational-financial-information/annual-reports",
    ]

    def discover(self) -> list:
        soup = self.fetch_html(self.hub_urls[0])
        docs = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = urljoin(self.hub_urls[0], a["href"])
            if not re.search(r"\.pdf(?:\?|#|$)", href.lower()):
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
                file_format="pdf",
                priority=priority,
                source_hub=self.hub_urls[0],
            ))

        return docs

    @staticmethod
    def _classify(title: str):
        t = title.lower()
        for pattern, doc_type, priority in TITLE_PATTERNS:
            if re.search(pattern, t):
                return doc_type, priority
        return None, None
