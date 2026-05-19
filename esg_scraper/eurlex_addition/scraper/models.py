"""Document data model and controlled vocabulary for ESG scraper."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# Controlled vocabulary for doc_type — keep this in sync with downstream code
DOC_TYPES = {
    "urd",                    # Universal Registration Document (FR companies)
    "annual_report",          # Combined annual + ESRS statement
    "sustainability_report",  # Standalone ESRS / non-financial information
    "integrated_report",      # IIRC-style integrated report
    "climate_report",         # Climate / TCFD-specific
    "progress_report",        # TotalEnergies-style annual progress
    "esg_tracker",            # Schneider quarterly SSI
    "esg_databook",           # Quantitative data (often .xlsx)
    "framework_disclosure",   # CDP, GRI, SASB, WEF, TCFD
    "thematic_report",        # Single topic (palm oil, methane, biodiversity)
    "vigilance_plan",         # French Duty of Vigilance Law
    "policy",                 # Code of conduct, sector policies
    "regulation",             # EU regulations, framework standards (ESRS, GHG Protocol, etc.)
    "other",
}

# Priority levels for RAG indexing
PRIORITY_PRIMARY = 1    # core ESRS doc — always index
PRIORITY_SECONDARY = 2  # supplementary — index if budget allows
PRIORITY_ARCHIVE = 3    # download but don't index by default


@dataclass
class Document:
    """A single document discovered on a company's site."""
    company: str
    url: str
    title: str
    doc_type: str
    fiscal_year: Optional[int] = None
    publication_year: Optional[int] = None
    language: str = "en"
    file_format: str = "pdf"
    priority: int = PRIORITY_SECONDARY
    source_hub: str = ""
    notes: str = ""

    # Filled after download
    file_path: Optional[Path] = None
    file_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    downloaded_at: Optional[datetime] = None

    def __post_init__(self):
        if self.doc_type not in DOC_TYPES:
            raise ValueError(
                f"Unknown doc_type {self.doc_type!r}. "
                f"Must be one of {sorted(DOC_TYPES)}"
            )

    def to_manifest_row(self) -> dict:
        """Flatten to a CSV-friendly dict."""
        d = asdict(self)
        d["file_path"] = str(self.file_path) if self.file_path else ""
        d["downloaded_at"] = (
            self.downloaded_at.isoformat() if self.downloaded_at else ""
        )
        return d
