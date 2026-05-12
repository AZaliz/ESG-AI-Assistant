"""Schneider Electric extractor.

Two cadences:
  - Annual: Universal Registration Document (URD) on the IR site
  - Quarterly: Sustainability Impact tracker, e.g.:
      schneider-sustainability-impact-q3-2025-results.pdf

Note: 2021–2025 SSI program ended; new 2030 roadmap launched late 2025.
Doc structure may shift in 2026 — be prepared to update patterns.
"""
import logging
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY, PRIORITY_ARCHIVE

logger = logging.getLogger(__name__)


# (regex on URL, doc_type, priority). Captures quarter for trackers.
URL_PATTERNS = [
    (r"schneider-sustainability-impact-q(\d)-(\d{4})", "esg_tracker",   PRIORITY_PRIMARY),
    (r"sustainable.development.report",                "urd",           PRIORITY_PRIMARY),
    (r"universal.registration",                        "urd",           PRIORITY_PRIMARY),
    (r"climate.transition",                            "climate_report",PRIORITY_PRIMARY),
    (r"sustainability.report",                         "sustainability_report", PRIORITY_PRIMARY),
    (r"vigilance",                                     "vigilance_plan",PRIORITY_PRIMARY),
    (r"human.rights",                                  "thematic_report",PRIORITY_SECONDARY),
    (r"supplier.code",                                 "policy",        PRIORITY_SECONDARY),
    (r"diversity",                                     "policy",        PRIORITY_SECONDARY),
]


HARDCODED = [
    {
        "url": "https://www.se.com/ww/en/assets/564/document/513141/2024-sustainability-report.pdf",
        "title": "2024 Sustainable Development Report",
        "doc_type": "urd",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
]


class SchneiderScraper(BaseScraper):
    company = "Schneider Electric"
    hub_urls = [
        "https://www.se.com/ww/en/about-us/sustainability/sustainability-reports/",
        "https://www.se.com/ww/en/about-us/investor-relations/regulatory-information/annual-reports/",
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

        # Discovery on both hubs
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
                if href in seen:
                    continue
                seen.add(href)

                title = a.get_text(strip=True)
                if not title or len(title) < 5:
                    title = href.rsplit("/", 1)[-1]

                doc_type, priority, quarter = self._classify(href)
                if doc_type is None:
                    continue

                year = self.extract_year(href, title)

                notes = f"q{quarter}" if quarter else ""

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
                    notes=notes,
                ))

        return docs

    @staticmethod
    def _classify(url: str):
        """Return (doc_type, priority, quarter_or_None)."""
        for pattern, doc_type, priority in URL_PATTERNS:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                # First pattern captures quarter
                quarter = m.group(1) if m.groups() and len(m.groups()) >= 1 else None
                # Only the SSI-tracker pattern actually has groups
                if "sustainability-impact-q" not in url:
                    quarter = None
                return doc_type, priority, quarter
        return None, None, None
