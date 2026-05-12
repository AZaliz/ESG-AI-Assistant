"""TotalEnergies extractor.

Hub page lists every report with stable URL slug patterns:
  - .../totalenergies_universal-registration-document-{YEAR}_{PUB}_en.pdf
  - .../totalenergies_sustainability-climate-{YEAR}-progress-report_{PUB}_en.pdf
  - .../totalenergies_esg-databook-{YEAR}_{PUB}_fr_en.xlsx
  - .../totalenergies_vpshr-annual-report_{YEAR}.pdf
"""
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY


# (regex, doc_type, priority). Order matters — first match wins.
URL_PATTERNS = [
    (r"universal-registration-document",         "urd",                  PRIORITY_PRIMARY),
    (r"sustainability-climate.*progress-report", "progress_report",      PRIORITY_PRIMARY),
    (r"climate-transition-plan",                 "climate_report",       PRIORITY_PRIMARY),
    (r"climate-report",                          "climate_report",       PRIORITY_PRIMARY),
    (r"esg-databook",                            "esg_databook",         PRIORITY_PRIMARY),
    (r"sustainability-statement",                "sustainability_report",PRIORITY_PRIMARY),
    (r"vpshr-annual-report",                     "thematic_report",      PRIORITY_SECONDARY),
    (r"tax-transparency-report",                 "thematic_report",      PRIORITY_SECONDARY),
    (r"human-rights-briefing",                   "thematic_report",      PRIORITY_SECONDARY),
    (r"code-of-conduct",                         "policy",               PRIORITY_SECONDARY),
    (r"conflict-minerals-report",                "thematic_report",      PRIORITY_SECONDARY),
    (r"reporting-(?:gri|sasb|wef)",              "framework_disclosure", PRIORITY_SECONDARY),
    (r"cdp-corporate-questionnaire",             "framework_disclosure", PRIORITY_SECONDARY),
    (r"sdg-reporting",                           "framework_disclosure", PRIORITY_SECONDARY),
    (r"index-gri",                               "framework_disclosure", PRIORITY_SECONDARY),
    (r"vigilance",                               "vigilance_plan",       PRIORITY_PRIMARY),
    (r"diversity-roadmap",                       "policy",               PRIORITY_SECONDARY),
    (r"gender-equality",                         "thematic_report",      PRIORITY_SECONDARY),
]


class TotalEnergiesScraper(BaseScraper):
    company = "TotalEnergies"
    hub_urls = [
        "https://totalenergies.com/sustainability/our-approach/esg-documentation",
    ]

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

            doc_type, priority = self._classify(href)
            if doc_type is None:
                continue

            year = self.extract_year(href, a.get_text(strip=True))
            title = a.get_text(strip=True) or href.rsplit("/", 1)[-1]

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

        return docs

    @staticmethod
    def _classify(url: str):
        for pattern, doc_type, priority in URL_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return doc_type, priority
        return None, None
