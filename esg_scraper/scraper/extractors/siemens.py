"""Siemens extractor.

Hardest target: PDFs hosted on assets.new.siemens.com with UUID-based URLs
(e.g. /uuid:dea0c623-1ae9-4ef0-a69a-31d8eb7b39fb/sustainability-statement.pdf).
The UUID changes every release — can't predict next year's URL.

Strategy:
  - Hardcode known essentials (Sustainability Statement)
  - Scrape the hub for any link to assets.new.siemens.com
  - Year extraction relies on link text or surrounding HTML, NOT the URL
"""
import logging
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY

logger = logging.getLogger(__name__)


HARDCODED = [
    {
        "url": "https://assets.new.siemens.com/siemens/assets/api/uuid:dea0c623-1ae9-4ef0-a69a-31d8eb7b39fb/sustainability-statement.pdf",
        "title": "Siemens Sustainability Statement (FY2024)",
        "doc_type": "sustainability_report",
        "fiscal_year": 2024,
        "publication_year": 2024,  # Siemens FY ends Sept 30
        "priority": PRIORITY_PRIMARY,
    },
]


TITLE_PATTERNS = [
    (r"sustainability statement|esrs",          "sustainability_report", PRIORITY_PRIMARY),
    (r"sustainability information",             "framework_disclosure",  PRIORITY_PRIMARY),  # Siemens-specific voluntary disclosure
    (r"sustainability report",                  "sustainability_report", PRIORITY_PRIMARY),
    (r"degree framework",                       "thematic_report",       PRIORITY_SECONDARY),
    (r"compliance.*report|business conduct",    "policy",                PRIORITY_SECONDARY),
    (r"climate report|tcfd",                    "climate_report",        PRIORITY_PRIMARY),
    (r"annual report",                          "annual_report",         PRIORITY_PRIMARY),
    (r"human rights",                           "thematic_report",       PRIORITY_SECONDARY),
    (r"supplier code",                          "policy",                PRIORITY_SECONDARY),
]


class SiemensScraper(BaseScraper):
    company = "Siemens"
    hub_urls = [
        "https://www.siemens.com/en-us/company/sustainability/reports-figures/",
    ]

    def discover(self) -> list:
        docs = []
        seen = set()

        # Hardcoded essentials
        for h in HARDCODED:
            seen.add(h["url"])
            docs.append(Document(
                company=self.company,
                source_hub="hardcoded",
                language="en", file_format="pdf",
                **h,
            ))

        # Discovery
        try:
            soup = self.fetch_html(self.hub_urls[0])
        except Exception as e:
            logger.warning(f"  [{self.company}] hub fetch failed for {self.hub_urls[0]}: {e}")
            return docs

        # Walk all links — also track current heading for year context
        current_year = None
        for el in soup.find_all(["h1", "h2", "h3", "a"]):
            if el.name in ("h1", "h2", "h3"):
                m = re.search(r"\b(20\d{2})\b", el.get_text())
                if m:
                    current_year = int(m.group(1))
                continue

            href = el.get("href")
            if not href:
                continue
            href = urljoin(self.hub_urls[0], href)
            if not re.search(r"\.pdf(?:\?|#|$)", href.lower()):
                continue
            if href in seen:
                continue
            # Siemens hosts on assets.new.siemens.com or siemens.com
            if not re.search(r"siemens\.com", href):
                continue
            seen.add(href)

            title = el.get_text(strip=True)
            if not title or len(title) < 5:
                continue

            doc_type, priority = self._classify(title)
            if doc_type is None:
                continue

            # URL has no year — rely on title and surrounding heading
            year = self.extract_year(title) or current_year

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
