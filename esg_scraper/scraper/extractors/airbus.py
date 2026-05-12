"""Airbus extractor.

Two source pages:
  1. Sustainability standards & performance hub (links framework disclosures)
  2. Hardcoded essentials (Pioneering publication + ESG Datasheet)

The ESG Datasheet uses unguessable slug URLs on mediaassets.airbus.com,
so we hardcode it. Update HARDCODED yearly.
"""
import logging
import re
from urllib.parse import urljoin

from ..base import BaseScraper
from ..models import Document, PRIORITY_PRIMARY, PRIORITY_SECONDARY

logger = logging.getLogger(__name__)


HARDCODED = [
    {
        "url": "https://www.airbus.com/sites/g/files/jlcbta136/files/2025-04/2025_Airbus_Pioneering_sustainable_aerospace_publication.pdf",
        "title": "2025 Pioneering Sustainable Aerospace",
        "doc_type": "sustainability_report",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
    {
        "url": "https://mediaassets.airbus.com/pm_38_711_711991-us758j55ai.pdf",
        "title": "2025 ESG Datasheet (FY2024)",
        "doc_type": "esg_databook",
        "fiscal_year": 2024,
        "publication_year": 2025,
        "priority": PRIORITY_PRIMARY,
    },
]


class AirbusScraper(BaseScraper):
    company = "Airbus"
    hub_urls = [
        "https://www.airbus.com/en/sustainability/sustainability-standards-and-performance",
        # Real document hub — host of CDP/ESG datasheet/policies/codes of conduct
        "https://www.airbus.com/en/sustainability/sustainability-standards-and-performance/document-centre",
    ]

    def discover(self) -> list:
        docs = []
        seen = set()

        # Hardcoded essentials first
        for h in HARDCODED:
            seen.add(h["url"])
            docs.append(Document(
                company=self.company,
                source_hub="hardcoded",
                language="en", file_format="pdf",
                **h,
            ))

        for hub in self.hub_urls:
            try:
                soup = self.fetch_html(hub)
            except Exception as e:
                logger.warning(f"  [{self.company}] hub fetch failed for {hub}: {e}")
                continue  # hardcoded entries still work

            for a in soup.find_all("a", href=True):
                href = urljoin(hub, a["href"])
                if not re.search(r"\.pdf(?:\?|#|$)", href.lower()):
                    continue
                href_key = href.split("?")[0]
                if href_key in seen or href in seen:
                    continue
                seen.add(href_key)
                seen.add(href)

                # Most Airbus document-centre links have text "Download" — fall back to URL filename.
                # mediaassets.airbus.com encodes the real name in ?fileName=...
                title = a.get_text(strip=True)
                if not title or title.lower() == "download" or len(title) < 5:
                    title = self._title_from_url(href)
                if not title:
                    continue

                # Match patterns against title + URL (normalized)
                matchable = (title + " " + href).lower().replace("-", " ").replace("_", " ")

                if not re.search(
                    r"tcfd|gri|sasb|cdp|datasheet|sustainability|annual report|"
                    r"climate|carbon reduction|"
                    r"human rights|modern slavery|"
                    r"code of conduct|supplier code|"
                    r"environmental policy|tax strategy|"
                    r"diversity|gender pay|inclusion|esg",
                    matchable,
                ):
                    continue

                doc_type = (
                    "sustainability_report" if re.search(r"sustainability|pioneering", matchable)
                    else "esg_databook" if re.search(r"datasheet", matchable)
                    else "climate_report" if re.search(r"climate|carbon reduction|cdp", matchable)
                    else "policy" if re.search(r"code of conduct|supplier code|tax strategy|environmental policy|responsible mineral", matchable)
                    else "thematic_report" if re.search(r"human rights|modern slavery|gender pay|diversity|inclusion", matchable)
                    else "framework_disclosure"
                )

                docs.append(Document(
                    company=self.company,
                    url=href,
                    title=title[:200],
                    doc_type=doc_type,
                    fiscal_year=self.extract_year(title, href),
                    language="en",
                    file_format="pdf",
                    priority=PRIORITY_SECONDARY,
                    source_hub=hub,
                ))

        return docs

    @staticmethod
    def _title_from_url(url: str) -> str:
        """Build a readable title from the URL filename when the <a> text is empty/'Download'."""
        # mediaassets.airbus.com puts the real name in ?fileName=...
        m = re.search(r"fileName=([^&]+?)(?:\.pdf)?(?:&|$)", url, re.I)
        if m:
            return m.group(1).replace("-", " ").replace("_", " ")
        slug = url.split("?")[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return slug.replace("-", " ").replace("_", " ").replace("%20", " ")
