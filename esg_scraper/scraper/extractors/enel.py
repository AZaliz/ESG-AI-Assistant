"""Enel extractor.

Hub: https://www.enel.com/investors/sustainability

Every download link follows a stable AEM pattern:
  https://www.enel.com/content/dam/enel-com/documenti/investitori/
    {informazioni-finanziarie|sostenibilita}/{YEAR}/{slug}.pdf

The visible link text is always "Download" — the descriptive document name
lives in the link's title attribute as 'Download {DOC NAME}'.
Adjacent <h4> heading also carries the doc name; we use the title attribute
first and fall back to the heading.

Document type vocabulary:
  - Integrated Annual Report      → integrated_report (post-2024 = CSRD primary)
  - Sustainability Report          → sustainability_report
  - Consolidated Non-financial Statement (NFS, pre-CSRD predecessor) → sustainability_report
  - GHG Inventory                  → esg_databook
  - Climate Policy Advocacy        → climate_report
  - Net-Zero / Zero Emissions Amb. → climate_report
  - ESG Focus / ESG Supplement     → framework_disclosure
  - Green Bond Report              → framework_disclosure
  - European Taxonomy / Article 8  → framework_disclosure
  - Tax Transparency               → thematic_report
  - Managing Human Rights          → thematic_report
  - Just & inclusive transition    → thematic_report
  - Circular Economy               → thematic_report
  - Materiality analysis           → thematic_report
  - Half-year disclosures          → sustainability_report (SECONDARY)
  - Policy docs (env, H&S)         → policy
"""
import logging
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY, PRIORITY_ARCHIVE

logger = logging.getLogger(__name__)


# (regex, doc_type, priority). Order matters — first match wins.
# Patterns are checked against the document title (after stripping "Download ").
TITLE_PATTERNS = [
    (r"integrated annual report",                       "integrated_report",     PRIORITY_PRIMARY),
    (r"sustainability report.*non.financial statement", "sustainability_report", PRIORITY_PRIMARY),
    (r"sustainability report",                          "sustainability_report", PRIORITY_PRIMARY),
    (r"consolidated non.financial statement|^nfs",      "sustainability_report", PRIORITY_PRIMARY),
    (r"sustainability statement",                       "sustainability_report", PRIORITY_PRIMARY),
    (r"sustainability plan",                            "sustainability_report", PRIORITY_SECONDARY),
    (r"half.year sustainability",                       "sustainability_report", PRIORITY_SECONDARY),
    (r"environmental report",                           "sustainability_report", PRIORITY_PRIMARY),
    (r"(sustainability|environmental) country overview",  "sustainability_report", PRIORITY_SECONDARY),

    (r"ghg inventory",                                  "esg_databook",          PRIORITY_PRIMARY),
    (r"additional esg key performance indicators",      "esg_databook",          PRIORITY_PRIMARY),

    (r"climate policy advocacy|engagement.*climate.policy", "climate_report",    PRIORITY_PRIMARY),
    (r"net.zero|zero emissions ambition|path to net.zero",  "climate_report",    PRIORITY_PRIMARY),
    (r"commitment to (the )?fight against climate change",  "climate_report",    PRIORITY_PRIMARY),

    (r"esg focus for investors",                        "framework_disclosure",  PRIORITY_SECONDARY),
    (r"esg supplement",                                 "framework_disclosure",  PRIORITY_SECONDARY),
    (r"green bond report",                              "framework_disclosure",  PRIORITY_SECONDARY),
    (r"european taxonomy|environmentally sustainable.*economic activities|article 8.*regulation", "framework_disclosure", PRIORITY_PRIMARY),
    (r"cdp",                                            "framework_disclosure",  PRIORITY_SECONDARY),
    (r"at a glance",                                    "framework_disclosure",  PRIORITY_ARCHIVE),
    (r"executive summary",                              "framework_disclosure",  PRIORITY_ARCHIVE),

    (r"managing human rights|approach.*human rights|responsible business conduct", "thematic_report", PRIORITY_SECONDARY),
    (r"tax transparency|total tax contribution|tax.*approach", "thematic_report", PRIORITY_SECONDARY),
    (r"just.*inclusive transition",                     "thematic_report",       PRIORITY_SECONDARY),
    (r"circular economy",                               "thematic_report",       PRIORITY_SECONDARY),
    (r"materiality analysis",                           "thematic_report",       PRIORITY_SECONDARY),
    (r"communities and value sharing",                  "thematic_report",       PRIORITY_SECONDARY),
    (r"low.carbon|low carbon growth",                   "thematic_report",       PRIORITY_SECONDARY),
    (r"long term sustainable growth",                   "thematic_report",       PRIORITY_ARCHIVE),

    (r"environmental policy",                           "policy",                PRIORITY_SECONDARY),
    (r"health and safety policy",                       "policy",                PRIORITY_SECONDARY),
]


# Critical docs hardcoded so the scraper produces useful output even if the
# hub structure changes. Update yearly.
HARDCODED = [
    {
        "url": "https://www.enel.com/content/dam/enel-com/documenti/investitori/informazioni-finanziarie/2024/annuali/en/integrated-annual-report_2024.pdf",
        "title": "Integrated Annual Report 2024",
        "doc_type": "integrated_report",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
    {
        "url": "https://www.enel.com/content/dam/enel-com/documenti/investitori/sostenibilita/2024/ghg-inventory-2024.pdf",
        "title": "2024 GHG Inventory",
        "doc_type": "esg_databook",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
    {
        "url": "https://www.enel.com/content/dam/enel-com/documenti/investitori/sostenibilita/2024/climate-policy-advocacy-report-2024.pdf",
        "title": "Climate Policy Advocacy Report 2024",
        "doc_type": "climate_report",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
]


def _clean_title(raw: str) -> str:
    """Strip the 'Download ' prefix and any trailing quotation/whitespace."""
    t = (raw or "").strip()
    t = re.sub(r"^Download\s+", "", t, flags=re.IGNORECASE)
    t = t.strip(' "\'')
    return t


class EnelScraper(BaseScraper):
    company = "Enel"
    hub_urls = ["https://www.enel.com/investors/sustainability"]

    def discover(self) -> list:
        docs = []
        seen = set()

        # 1. Hardcoded essentials — these run even if the hub fetch fails
        for h in HARDCODED:
            seen.add(h["url"])
            docs.append(Document(
                company=self.company,
                source_hub="hardcoded",
                language="en", file_format="pdf",
                **h,
            ))

        # 2. Hub discovery
        try:
            soup = self.fetch_html(self.hub_urls[0])
        except Exception as e:
            logger.warning(f"  [{self.company}] hub fetch failed for {self.hub_urls[0]}: {e}")
            return docs  # hardcoded entries still work

        for a in soup.find_all("a", href=True):
            href = urljoin(self.hub_urls[0], a["href"])
            if not re.search(r"\.pdf(?:\?|#|$)", href.lower()):
                continue
            if "/content/dam/" not in href:
                # Filter out random non-document PDFs (e.g., from navigation)
                continue
            if href in seen:
                continue
            seen.add(href)

            # Title comes from the `title="Download X"` attribute. Fall back to
            # the preceding heading, then to the URL slug.
            title = _clean_title(a.get("title", ""))
            if not title or title.lower() == "download":
                heading = a.find_previous(["h2", "h3", "h4", "h5"])
                if heading:
                    title = heading.get_text(strip=True)
            if not title:
                slug = href.rsplit("/", 1)[-1]
                title = slug.replace("_", " ").replace("-", " ").replace(".pdf", "")
            if len(title) < 5:
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