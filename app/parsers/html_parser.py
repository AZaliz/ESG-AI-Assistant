"""HTML visible-text extraction."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.models import ParseResult
from app.utils import visible_text_chunks


def parse_html(path: Path) -> ParseResult:
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return ParseResult(parser_name="beautifulsoup", status="failed", notes=f"Unable to read HTML: {exc}")

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    text = visible_text_chunks(soup.stripped_strings)
    if not text:
        return ParseResult(
            parser_name="beautifulsoup",
            status="partial",
            notes="HTML downloaded but visible text extraction was empty.",
            extracted_title=title,
        )

    return ParseResult(
        parser_name="beautifulsoup",
        status="success",
        text=text,
        extracted_title=title,
    )
