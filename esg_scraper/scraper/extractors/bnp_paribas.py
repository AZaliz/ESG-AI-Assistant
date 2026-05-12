"""BNP Paribas extractor.

CSR documents hub lists thematic reports. The full Integrated Report and URD
live on invest.bnpparibas behind JS — we hardcode them.
"""
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY, PRIORITY_ARCHIVE


TITLE_PATTERNS = [
    (r"climate report",                      "climate_report",       PRIORITY_PRIMARY),
    (r"climate analytics|alignment report",  "climate_report",       PRIORITY_PRIMARY),
    (r"environmental framework",             "policy",               PRIORITY_SECONDARY),
    (r"engagement manifesto",                "policy",               PRIORITY_SECONDARY),
    (r"human rights",                        "thematic_report",      PRIORITY_SECONDARY),
    (r"biodiversity",                        "thematic_report",      PRIORITY_SECONDARY),
    (r"vigilance plan",                      "vigilance_plan",       PRIORITY_PRIMARY),
    (r"modern slavery",                      "thematic_report",      PRIORITY_SECONDARY),
    (r"prb reporting",                       "framework_disclosure", PRIORITY_SECONDARY),
    (r"cdp",                                 "framework_disclosure", PRIORITY_SECONDARY),
    (r"equator principles",                  "framework_disclosure", PRIORITY_SECONDARY),
    (r"direct environmental impact",         "thematic_report",      PRIORITY_SECONDARY),
    (r"supporting.*transition.*csr|csr achievements","sustainability_report", PRIORITY_PRIMARY),
    (r"urd|universal registration",          "urd",                  PRIORITY_PRIMARY),
    (r"voluntary carbon",                    "policy",               PRIORITY_SECONDARY),
    (r"perspectives",                        "thematic_report",      PRIORITY_ARCHIVE),
    (r"microfinance",                        "thematic_report",      PRIORITY_SECONDARY),
    (r"just transition observatory",         "thematic_report",      PRIORITY_SECONDARY),
    (r"integrated report",                   "integrated_report",    PRIORITY_PRIMARY),
    (r"sustainable.{0,8}sourcing charter",   "policy",               PRIORITY_SECONDARY),
    (r"responsible business principles",     "policy",               PRIORITY_SECONDARY),
    (r"iso.*certifications",                 "framework_disclosure", PRIORITY_ARCHIVE),
    (r"stakeholder",                         "thematic_report",      PRIORITY_SECONDARY),
    (r"blue economy",                        "thematic_report",      PRIORITY_SECONDARY),
    (r"sustainable finance",                 "thematic_report",      PRIORITY_SECONDARY),
]

# Critical docs not on the CSR hub
HARDCODED = [
    {
        "url": "https://cdn-group.bnpparibas.com/uploads/file/bnp_paribas_integrated_report_2024_en_bd_1.pdf",
        "title": "2024 Integrated Report",
        "doc_type": "integrated_report",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
]


class BNPParibasScraper(BaseScraper):
    company = "BNP Paribas"
    hub_urls = ["https://group.bnpparibas/en/group/publications/csr-documents"]

    def discover(self) -> list:
        soup = self.fetch_html(self.hub_urls[0])
        docs = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = urljoin(self.hub_urls[0], a["href"])
            if not re.search(r"\.(pdf|xlsx)(?:\?|#|$)", href.lower()):
                continue
            if href in seen:
                continue
            seen.add(href)

            heading = a.find(["h3", "h4"])
            title = (heading.get_text(strip=True) if heading
                     else a.get_text(strip=True))
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
                source_hub=self.hub_urls[0],
            ))

        # Add hardcoded essentials
        for extra in HARDCODED:
            if extra["url"] in seen:
                continue
            docs.append(Document(
                company=self.company,
                source_hub="hardcoded",
                language="en", file_format="pdf",
                **extra,
            ))

        return docs

    @staticmethod
    def _classify(title: str):
        t = title.lower()
        for pattern, doc_type, priority in TITLE_PATTERNS:
            if re.search(pattern, t):
                return doc_type, priority
        return None, None
