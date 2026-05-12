"""Volkswagen extractor.

From FY2024, Volkswagen folded sustainability into the Annual Report and the
ESRS Sustainability Statement is hosted on a year-specific subdomain
(annualreport2024.volkswagen-group.com). Older standalone Group Sustainability
Reports are on the main reporting hub.

Strategy: hardcode the latest year-specific subdomain URLs + scrape the hub
for older standalone PDFs.
"""
import logging
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY

logger = logging.getLogger(__name__)


HARDCODED = [
    {
        "url": "https://annualreport2024.volkswagen-group.com/_assets/downloads/esrs-sustainability-report-vw-ar24.pdf",
        "title": "2024 ESRS Sustainability Report",
        "doc_type": "sustainability_report",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
    # When FY2025 drops, add: annualreport2025.volkswagen-group.com/...
]


TITLE_PATTERNS = [
    (r"sustainability report",                            "sustainability_report", PRIORITY_PRIMARY),
    (r"esrs|sustainability statement",                    "sustainability_report", PRIORITY_PRIMARY),
    (r"annual report",                                    "annual_report",         PRIORITY_PRIMARY),
    (r"non.financial group report|nfgr",                  "sustainability_report", PRIORITY_PRIMARY),
    (r"climate report|climate review|climate change|tcfd","climate_report",        PRIORITY_PRIMARY),
    (r"raw materials? report|responsible.raw",            "thematic_report",       PRIORITY_PRIMARY),
    (r"compliance report|integrity",                      "policy",                PRIORITY_SECONDARY),
    (r"diversity|inclusion",                              "policy",                PRIORITY_SECONDARY),
    (r"supply chain|due diligence",                       "thematic_report",       PRIORITY_SECONDARY),
    (r"financial report|consolidated|financial statement","annual_report",         PRIORITY_SECONDARY),
    (r"\bgri\b|content index",                            "framework_disclosure",  PRIORITY_SECONDARY),
    (r"\bsasb\b",                                         "framework_disclosure",  PRIORITY_SECONDARY),
    (r"\bcdp\b|water security",                           "framework_disclosure",  PRIORITY_SECONDARY),
]


class VolkswagenScraper(BaseScraper):
    company = "Volkswagen"
    hub_urls = [
        "https://www.volkswagen-group.com/en/reporting-15808",
        "https://www.volkswagen-group.com/en/financial-reports-18134",
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

        # Discovery on hub pages
        for hub in self.hub_urls:
            try:
                soup = self.fetch_html(hub)
            except Exception as e:
                logger.warning(f"  [{self.company}] hub fetch failed for {hub}: {e}")
                continue

            for a in soup.find_all("a", href=True):
                href = urljoin(hub, a["href"])
                if not re.search(r"\.pdf(?:\?|#|$)", href.lower()):
                    continue
                # Dedup by URL path: VW exposes each PDF three times with
                # different query strings (?timestamp, &disposition=attachment)
                href_key = href.split("?")[0]
                if href_key in seen:
                    continue
                seen.add(href_key)

                title = a.get_text(strip=True)
                if not title or len(title) < 5:
                    # Many <a> tags are icon-only; derive a title from the filename
                    slug = href_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    title = slug.replace("-", " ").replace("_", " ")
                if not title or len(title) < 5:
                    continue

                # Match against title + URL (normalized) so URL-slug keywords count
                matchable = (title + " " + href).lower().replace("-", " ").replace("_", " ")
                doc_type, priority = self._classify(matchable)
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
