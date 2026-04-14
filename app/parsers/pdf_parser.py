"""PDF text extraction using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz

from app.models import ParseResult


def parse_pdf(path: Path) -> ParseResult:
    try:
        document = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(parser_name="pymupdf", status="failed", notes=f"Unable to open PDF: {exc}")

    texts: list[str] = []
    title = document.metadata.get("title") if document.metadata else None
    for page in document:
        text = page.get_text("text").strip()
        if text:
            texts.append(text)

    combined = "\n\n".join(texts).strip()
    if not combined:
        return ParseResult(
            parser_name="pymupdf",
            status="partial",
            notes="PDF opened but yielded no extractable text.",
            page_count=document.page_count,
            extracted_title=title,
        )

    return ParseResult(
        parser_name="pymupdf",
        status="success",
        text=combined,
        notes="",
        page_count=document.page_count,
        extracted_title=title,
    )
