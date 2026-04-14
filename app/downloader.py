"""Download discovered documents with retries and local deduplication."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models import CandidateLink
from app.utils import RAW_DIR, filename_from_url, normalize_url, sha256_file, slugify


LOGGER = logging.getLogger(__name__)


class Downloader:
    def __init__(self, session: requests.Session) -> None:
        self.session = session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def download(self, company_name: str, candidate: CandidateLink) -> tuple[Path, str, str, str]:
        response = self.session.get(candidate.url, timeout=60, stream=True)
        response.raise_for_status()
        final_url = normalize_url(response.url)
        content_type = response.headers.get("Content-Type", "application/octet-stream").split(";")[0].strip()
        company_dir = RAW_DIR / slugify(company_name)
        company_dir.mkdir(parents=True, exist_ok=True)

        filename = self._resolve_filename(candidate, final_url, content_type)
        path = company_dir / filename
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    handle.write(chunk)

        file_hash = sha256_file(path)
        return path, final_url, content_type, file_hash

    @staticmethod
    def _resolve_filename(candidate: CandidateLink, final_url: str, content_type: str) -> str:
        name = candidate.file_name or filename_from_url(final_url)
        suffix = Path(name).suffix.lower()
        if not suffix:
            if content_type == "application/pdf":
                suffix = ".pdf"
            elif "html" in content_type:
                suffix = ".html"
            else:
                suffix = ".bin"
            name = f"{name}{suffix}"
        return Path(name).name
