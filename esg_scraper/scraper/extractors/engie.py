"""Engie extractor.

Hub page groups documents under year headings (h2/h3). Walk the DOM in
document order and track the active publication year as we go:

  <h2>2026</h2>
    <a href="...ENGIE_DEU_2025_UK.pdf">2025 Universal Registration Document</a>
    <a href="...ESG @ ENGIE 2025.pdf">ESG at ENGIE slide deck</a>
  <h2>2025</h2>
    ...

The heading year is publication_year. Title may contain fiscal_year.
"""
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY


TITLE_PATTERNS = [
    (r"universal registration document",       "urd",                  PRIORITY_PRIMARY),
    (r"esg.*databook|data\s*book",             "esg_databook",         PRIORITY_PRIMARY),
    (r"esg.{0,5}@.{0,5}engie|esg at engie",    "sustainability_report",PRIORITY_PRIMARY),
    (r"sustainability statement",              "sustainability_report",PRIORITY_PRIMARY),
    (r"integrated report",                     "integrated_report",    PRIORITY_PRIMARY),
    (r"climate notebook|tcdf|tcfd",            "climate_report",       PRIORITY_PRIMARY),
    (r"sustainability book",                   "sustainability_report",PRIORITY_PRIMARY),
    (r"vigilance plan",                        "vigilance_plan",       PRIORITY_PRIMARY),
    (r"biodiversity notebook",                 "thematic_report",      PRIORITY_SECONDARY),
    (r"just transition|stakeholder",           "thematic_report",      PRIORITY_SECONDARY),
    (r"bloomberg|gei",                         "framework_disclosure", PRIORITY_SECONDARY),
]


class EngieScraper(BaseScraper):
    company = "Engie"
    hub_urls = ["https://www.engie.com/en/investors/ESG"]

    def discover(self) -> list:
        soup = self.fetch_html(self.hub_urls[0])
        docs = []
        seen = set()
        current_pub_year = None

        # Walk all h2/h3/a in document order
        for el in soup.find_all(["h2", "h3", "a"]):
            if el.name in ("h2", "h3"):
                m = re.search(r"\b(20\d{2})\b", el.get_text())
                if m:
                    current_pub_year = int(m.group(1))
                continue

            href = el.get("href")
            if not href:
                continue
            href = urljoin(self.hub_urls[0], href)
            if not re.search(r"\.(pdf|xlsx)(?:\?|#|$)", href.lower()):
                continue
            if href in seen:
                continue
            seen.add(href)

            title = el.get_text(strip=True) or href.rsplit("/", 1)[-1]
            doc_type, priority = self._classify(title)
            if doc_type is None:
                continue

            fiscal = self.extract_year(title, href)
            # Sanity: fiscal_year should not be after publication_year
            if fiscal and current_pub_year and fiscal > current_pub_year:
                fiscal = None

            docs.append(Document(
                company=self.company,
                url=href,
                title=title[:200],
                doc_type=doc_type,
                fiscal_year=fiscal,
                publication_year=current_pub_year,
                language="en",
                file_format=self.file_format_from_url(href),
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
