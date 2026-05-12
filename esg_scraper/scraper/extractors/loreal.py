"""L'Oréal extractor.

The ESG hub on loreal.com links to URDs hosted on loreal-finance.com (different
subdomain). Policy documents (Climate Transition Plan, etc.) are on loreal.com
itself. Both must be allowed through.
"""
import logging
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY

logger = logging.getLogger(__name__)


TITLE_PATTERNS = [
    # \bdeu\b = Document d'Enregistrement Universel (French URD acronym in L'Oréal's filenames)
    (r"universal registration document|\bdeu\b|loreal.*annual report",  "urd",                  PRIORITY_PRIMARY),
    (r"climate transition plan|climate",         "climate_report",       PRIORITY_PRIMARY),
    (r"sustainability report|sustainability statement|sustainability reporting", "sustainability_report", PRIORITY_PRIMARY),
    (r"responsible advertising|marketing",       "policy",               PRIORITY_SECONDARY),
    (r"human rights",                            "thematic_report",      PRIORITY_SECONDARY),
    (r"vigilance",                               "vigilance_plan",       PRIORITY_PRIMARY),
    (r"diversity|inclusion|gender",              "policy",               PRIORITY_SECONDARY),
    (r"environmental product design|environment policy|water|forest|sustainable land|responsible.*sourcing", "policy", PRIORITY_SECONDARY),
    (r"code of ethics",                          "policy",               PRIORITY_SECONDARY),
    (r"esg performance|esg report|\bpai\b",      "framework_disclosure", PRIORITY_SECONDARY),
    (r"integrated report",                       "integrated_report",    PRIORITY_PRIMARY),
]

# Hub doesn't always link the latest URD — hardcode the prof's
HARDCODED = [
    {
        "url": "https://www.loreal-finance.com/system/files/2025-03/2024_Universal_Registration_Document_LOREAL.pdf",
        "title": "2024 Universal Registration Document",
        "doc_type": "urd",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
]


class LOrealScraper(BaseScraper):
    company = "L'Oréal"
    hub_urls = ["https://www.loreal.com/en/esg-performance/"]

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

        # Discovery on the ESG hub
        try:
            soup = self.fetch_html(self.hub_urls[0])
        except Exception as e:
            logger.warning(f"  [{self.company}] hub fetch failed for {self.hub_urls[0]}: {e}")
            return docs

        for a in soup.find_all("a", href=True):
            href = urljoin(self.hub_urls[0], a["href"])
            # Allow both loreal.com and loreal-finance.com
            if not re.search(r"\.pdf(?:\?|#|$)", href.lower()):
                continue
            if "loreal" not in href.lower():
                continue
            if href in seen:
                continue
            seen.add(href)

            title = a.get_text(strip=True)
            if not title or len(title) < 5:
                # Most L'Oréal PDFs have empty link text — derive from URL filename
                slug = href.split("?")[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
                title = slug.replace("-", " ").replace("_", " ")
            if not title:
                continue

            # Match patterns against title + URL (normalized) so we catch
            # links whose only descriptive text is in the URL slug
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
