"""Shared helpers for URL normalization, filesystem paths, and text cleanup."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlparse, urlunparse


LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TEXT_DIR = DATA_DIR / "text"
CANDIDATE_DIR = DATA_DIR / "candidates"
OUTPUT_DIR = ROOT_DIR / "outputs"
DB_PATH = DATA_DIR / "metadata.db"

EUROPEAN_COUNTRIES = {
    "Austria",
    "Belgium",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "United Kingdom",
    "Norway",
    "Iceland",
    "Liechtenstein",
}

TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "mc_cid",
    "mc_eid",
}


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, TEXT_DIR, CANDIDATE_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_url(url: str, base_url: str | None = None) -> str:
    resolved = urljoin(base_url, url) if base_url else url
    parsed = urlparse(resolved)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = quote(unquote(parsed.path or "/"), safe="/:@")
    query = "&".join(
        f"{quote(k, safe='')}={quote(v, safe='')}"
        for k, v in sorted(parse_qsl(parsed.query, keep_blank_values=False))
        if k not in TRACKING_QUERY_KEYS
    )
    return urlunparse((parsed.scheme or "https", netloc, path, "", query, ""))


def same_domain(url: str, domain: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    domain = domain.lower()
    return host == domain or host.endswith(f".{domain}")


def filename_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    if not path:
        return "document"
    return unquote(path.split("/")[-1]) or "document"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_years(*values: str) -> list[int]:
    years: list[int] = []
    for value in values:
        for match in re.findall(r"\b(20\d{2})\b", value or ""):
            year = int(match)
            if 2000 <= year <= 2100 and year not in years:
                years.append(year)
    return years


def closest_year(years: Iterable[int], reference_year: int) -> int | None:
    year_list = list(years)
    if not year_list:
        return None
    return sorted(year_list, key=lambda year: (abs(reference_year - year), -year))[0]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def visible_text_chunks(strings: Iterable[str]) -> str:
    cleaned = [normalize_whitespace(item) for item in strings]
    return "\n".join(item for item in cleaned if item)
