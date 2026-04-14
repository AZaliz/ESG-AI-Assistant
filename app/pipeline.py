"""End-to-end acquisition pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from app.candidate_ranker import rank_candidates
from app.company_resolver import CompanyResolver
from app.downloader import Downloader
from app.models import CompanyProfile, CompanySeed, DocumentRecord, ParseResult
from app.normalizer import build_document_record
from app.parsers.html_parser import parse_html
from app.parsers.pdf_parser import parse_pdf
from app.source_discovery import SourceDiscovery
from app.storage import Storage
from app.utils import TEXT_DIR, normalize_whitespace, slugify, write_text


LOGGER = logging.getLogger(__name__)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


class AcquisitionPipeline:
    def __init__(self, storage: Storage | None = None) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.storage = storage or Storage()
        self.resolver = CompanyResolver(self.session)
        self.discovery = SourceDiscovery(self.session)
        self.downloader = Downloader(self.session)

    def discover(self, seed: CompanySeed):
        profile = self.resolver.resolve(seed)
        candidates = self.discovery.discover(profile)
        ranked = rank_candidates(candidates, profile)
        self.storage.save_candidates(profile.canonical_name, ranked)
        return profile, ranked

    def fetch_best(self, seed: CompanySeed) -> tuple[CompanyProfile, DocumentRecord | None, list[str]]:
        profile, ranked = self.discover(seed)
        notes = list(profile.notes)
        if not ranked:
            notes.append("No report candidates found.")
            return profile, None, notes

        best = ranked[0]
        try:
            download_path, final_url, mime_type, file_hash = self.downloader.download(profile.canonical_name, best)
        except requests.RequestException as exc:
            notes.append(f"Download failed for best candidate: {exc}")
            return profile, None, notes

        parse_result, text_path = self._parse_download(profile.canonical_name, download_path, mime_type)
        record = build_document_record(
            profile=profile,
            candidate=best,
            final_url=final_url,
            mime_type=mime_type,
            download_path=download_path,
            text_path=text_path,
            file_hash=file_hash,
            parse_result=parse_result,
        )
        if notes:
            record.notes = " | ".join(item for item in [record.notes, *notes] if item)
        self.storage.save_document(record)
        return profile, record, notes

    def _parse_download(self, company_name: str, path: Path, mime_type: str) -> tuple[ParseResult, Path | None]:
        if mime_type == "application/pdf" or path.suffix.lower() == ".pdf":
            result = parse_pdf(path)
        else:
            result = parse_html(path)

        text_path: Path | None = None
        if result.text:
            text_path = TEXT_DIR / slugify(company_name) / f"{path.stem}.txt"
            write_text(text_path, result.text)
        return result, text_path
