"""EU Regulators & Standards extractor.

Unlike company extractors, this one has no hub to discover — all four documents
are pinned URLs. They're stable government/foundation sources that change at most
every few years (only when regulations are amended).

Documents added here become authority-3 ('definitional anchor') in the
source_authority hierarchy: the RAG prompt should cite them for *definitions*
of terms (what counts as Scope 3 Cat 5, what 'double materiality' means under
ESRS), NOT for facts about specific companies. The build_index.py
source_authority_for() function tags 'regulation' doc_type as authority 3
automatically.

Why these four (per the audit-grounded weighting doc):
  - ESRS — defines every term the assistant will be asked about
  - CSDDD — forthcoming supply-chain due diligence regime
  - GHG Protocol — the canonical reference for Scope 1/2/3 definitions
  - IEA NZE — the most-cited 1.5°C scenario in CSRD climate disclosures
"""
import re
from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY


# EUR-Lex serves regulations as PDFs at /legal-content/EN/TXT/PDF/?uri=CELEX:<id>
# CELEX number conventions:
#   3 = legislative acts (regulations, directives)
#   0 = consolidated text
#   L = directive, R = regulation
# Examples:
#   32024L1760     = Directive 2024/1760 (CSDDD), original
#   02023R2772-20231222 = Regulation 2023/2772 (ESRS), consolidated as of 2023-12-22
HARDCODED = [
    {
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02023R2772-20231222",
        "title": "ESRS — Commission Delegated Regulation (EU) 2023/2772 (consolidated)",
        "doc_type": "regulation",
        "fiscal_year": 2024,           # year ESRS first applied (FY2024 reports)
        "publication_year": 2023,
        "priority": PRIORITY_PRIMARY,
        "notes": (
            "European Sustainability Reporting Standards. 12 annexes (ESRS 1, 2, "
            "E1–E5, S1–S4, G1). Authority=3 (definitional anchor). "
            "If consolidated URL fails, fallback: ?uri=CELEX:32023R2772"
        ),
    },
    {
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024L1760",
        "title": "CSDDD — Directive (EU) 2024/1760 on Corporate Sustainability Due Diligence",
        "doc_type": "regulation",
        "fiscal_year": 2024,
        "publication_year": 2024,
        "priority": PRIORITY_PRIMARY,
        "notes": (
            "Adopted June 2024. Member-State transposition deadlines through 2028. "
            "Authority=3 (forthcoming legal regime for supply-chain due diligence)."
        ),
    },
    {
        "url": "https://ghgprotocol.org/sites/default/files/standards/ghg-protocol-revised.pdf",
        "title": "GHG Protocol Corporate Accounting and Reporting Standard (revised edition)",
        "doc_type": "regulation",
        "fiscal_year": 2004,
        "publication_year": 2004,
        "priority": PRIORITY_PRIMARY,
        "notes": (
            "WRI/WBCSD canonical reference for Scope 1/2/3 definitions. "
            "Cited by every CSRD report's E1 disclosure. Authority=3 (terminology anchor)."
        ),
    },
    {
        "url": (
            "https://iea.blob.core.windows.net/assets/"
            "deebef5d-0c34-4539-9d0c-10b13d840027/"
            "NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf"
        ),
        "title": "IEA Net Zero by 2050 — A Roadmap for the Global Energy Sector",
        "doc_type": "regulation",
        "fiscal_year": 2021,
        "publication_year": 2021,
        "priority": PRIORITY_PRIMARY,
        "notes": (
            "Index ONLY the executive summary (~pp.1–60). The full 224-page report "
            "is sectoral modeling detail that's noise for RAG. Filter in chunker. "
            "Authority=3 (climate scenario reference)."
        ),
    },
]


class EURLexScraper(BaseScraper):
    """Pulls EU regulations and framework standards from canonical URLs.

    No hub discovery — all four documents are pinned. Re-runs are cheap
    because BaseScraper.download() skips already-downloaded files via
    content-hash deduplication.
    """
    company = "EU Regulators"
    hub_urls: list = []   # explicit: no hubs to crawl

    def discover(self) -> list:
        docs = []
        for h in HARDCODED:
            docs.append(Document(
                company=self.company,
                source_hub="hardcoded",
                language="en",
                file_format="pdf",
                **h,
            ))
        return docs
