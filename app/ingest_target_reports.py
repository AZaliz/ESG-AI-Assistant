"""Download a curated set of ESG reports and store them in the local database."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from app.downloader import Downloader
from app.models import CandidateLink, CompanyProfile, CompanySeed
from app.normalizer import build_document_record
from app.parsers.pdf_parser import parse_pdf
from app.storage import Storage
from app.utils import TEXT_DIR, ensure_directories, filename_from_url, slugify, write_text


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_STEM = "target_esg_report_ingest"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class TargetReport:
    name: str
    pdf_url: str


TARGET_REPORTS: list[TargetReport] = [
    TargetReport(
        name="TotalEnergies",
        pdf_url="https://www.totalenergies.com/system/files/documents/totalenergies_sustainability-climate-2025-progress-report_2025_en.pdf",
    ),
    TargetReport(
        name="BNP Paribas",
        pdf_url="https://cdn-group.bnpparibas.com/uploads/file/bnp_paribas_integrated_report_2024_en_bd_1.pdf",
    ),
    TargetReport(
        name="Airbus",
        pdf_url="https://www.airbus.com/sites/g/files/jlcbta136/files/2025-04/2025_Airbus_Pioneering_sustainable_aerospace_publication.pdf",
    ),
    TargetReport(
        name="Danone",
        pdf_url="https://www.danone.com/content/dam/corp/global/danonecom/investors/en-all-publications/2025/registrationdocuments/urd2024accessibleversion.pdf",
    ),
    TargetReport(
        name="Engie",
        pdf_url="https://www.engie.com/sites/default/files/assets/documents/2025-04/20250416%20-Engie%20-2024%20ESG%20at%20ENGIE.pdf",
    ),
    TargetReport(
        name="Schneider Electric",
        pdf_url="https://www.se.com/ww/en/assets/564/document/527425/schneider-sustainability-impact-q3-2025-results.pdf",
    ),
    TargetReport(
        name="L'Oreal",
        pdf_url="https://www.loreal-finance.com/system/files/2025-03/2024_Universal_Registration_Document_LOREAL.pdf",
    ),
    TargetReport(
        name="Volkswagen",
        pdf_url="https://annualreport2024.volkswagen-group.com/_assets/downloads/esrs-sustainability-report-vw-ar24.pdf",
    ),
    TargetReport(
        name="Siemens",
        pdf_url="https://assets.new.siemens.com/siemens/assets/api/uuid:dea0c623-1ae9-4ef0-a69a-31d8eb7b39fb/sustainability-statement.pdf",
    ),
    TargetReport(
        name="Iberdrola",
        pdf_url="https://www.iberdrola.com/documents/20125/42388/IB_Sustainability_Report.pdf",
    ),
]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def _humanize_name_from_url(pdf_url: str) -> str:
    stem = Path(filename_from_url(pdf_url)).stem
    return stem.replace("_", " ").replace("-", " ").strip() or Path(urlparse(pdf_url).path).stem


def _build_profile(company_name: str) -> CompanyProfile:
    seed = CompanySeed(
        name=company_name,
        ranking_source="manual",
        ranking_url="manual",
    )
    return CompanyProfile(
        seed=seed,
        canonical_name=company_name,
        source_urls=["manual"],
        notes=["manual target ingestion"],
    )


def ingest_target_reports(*, output_stem: str = DEFAULT_OUTPUT_STEM) -> dict[str, Any]:
    ensure_directories()
    storage = Storage()
    session = _session()
    downloader = Downloader(session)

    results: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    for target in TARGET_REPORTS:
        profile = _build_profile(target.name)
        candidate = CandidateLink(
            company_name=target.name,
            url=target.pdf_url,
            source_page_url=target.pdf_url,
            source_type="issuer_pdf",
            anchor_text=_humanize_name_from_url(target.pdf_url),
            page_title=target.name,
            nearby_text="Manual ESG report ingestion",
            file_name=filename_from_url(target.pdf_url),
            mime_hint="application/pdf",
            score=1.0,
            document_type="unknown",
        )

        try:
            download_path, final_url, mime_type, file_hash = downloader.download(target.name, candidate)
            parse_result = parse_pdf(download_path)
            text_path: Path | None = None
            if parse_result.text:
                text_path = TEXT_DIR / slugify(target.name) / f"{download_path.stem}.txt"
                write_text(text_path, parse_result.text)

            record = build_document_record(
                profile=profile,
                candidate=candidate,
                final_url=final_url,
                mime_type=mime_type,
                download_path=download_path,
                text_path=text_path,
                file_hash=file_hash,
                parse_result=parse_result,
            )
            storage.save_document(record)
            success_count += 1
            results.append(
                {
                    "company_name": target.name,
                    "status": "success",
                    "title": record.title,
                    "report_year": record.report_year,
                    "source_url": record.source_url,
                    "final_url": record.final_url,
                    "mime_type": record.mime_type,
                    "download_path": record.download_path,
                    "text_path": record.text_path,
                    "file_hash": record.file_hash,
                    "parse_status": record.parse_status,
                    "notes": record.notes,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            LOGGER.exception("Failed to ingest %s", target.name)
            results.append(
                {
                    "company_name": target.name,
                    "status": "failed",
                    "source_url": target.pdf_url,
                    "error": str(exc),
                }
            )

    json_path, csv_path = storage.save_smoke_outputs(results, output_stem)
    return {
        "success_count": success_count,
        "failure_count": failure_count,
        "total": len(TARGET_REPORTS),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "results": results,
    }


def run_ingest_target_reports(args: Any | None = None) -> int:
    output_stem = DEFAULT_OUTPUT_STEM if args is None else getattr(args, "output_stem", DEFAULT_OUTPUT_STEM)
    payload = ingest_target_reports(output_stem=output_stem)
    print(
        f"Ingested {payload['success_count']}/{payload['total']} reports into {payload['json_path']} and {payload['csv_path']}"
    )
    if payload["failure_count"]:
        print("Failures:")
        for result in payload["results"]:
            if result.get("status") == "failed":
                print(f"- {result['company_name']}: {result['error']}")
        return 1
    return 0