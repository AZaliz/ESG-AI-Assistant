"""Document parser: extract text AND tables from PDFs and XLSX, preserving page anchors.

The key audit finding was that ~75% of PARTIAL findings are figures locked in tables
that text-only extraction can't reconstruct. This module solves that by running
pdfplumber's table detection alongside text extraction, then handing both streams
to the normalizer.

Output schema (one record per page):
    {
        "page": int,
        "text": str,              # narrative text (tables stripped out)
        "tables": [               # extracted tables for this page
            {
                "table_id": str,
                "bbox": (x0, y0, x1, y1),
                "rows": [[cell, cell, ...], ...],  # raw cell text
                "header_row": [cell, ...] | None,  # auto-detected
            },
            ...
        ],
    }
"""
from __future__ import annotations

# --- UTF-8 console bootstrap: Windows wraps stdout in cp1252, which can't
# encode →, —, É (L'Oréal), etc. Force UTF-8 so output never crashes. ---
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pdfplumber

logger = logging.getLogger(__name__)


# pdfplumber table-detection settings tuned for corporate reports.
# These reports tend to use ruled tables (visible lines) for ESRS data,
# which "lines" strategy handles well. Fall back to "text" for borderless tables.
DEFAULT_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
}

FALLBACK_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 4,
    "intersection_x_tolerance": 8,
    "intersection_y_tolerance": 8,
}


@dataclass
class ExtractedTable:
    """A single table found on one page."""
    table_id: str
    page: int
    bbox: tuple
    rows: list[list[str]]
    header_row: list[str] | None = None

    def is_valid(self) -> bool:
        """Tables with too few cells or all-empty rows are usually garbage."""
        if not self.rows or len(self.rows) < 2:
            return False
        # Require at least 4 non-empty cells across the whole table
        non_empty = sum(
            1 for row in self.rows for c in row if c and c.strip()
        )
        if non_empty < 4:
            return False
        # Reject single-column tables (usually layout artifacts)
        max_cols = max(len(r) for r in self.rows)
        return max_cols >= 2


@dataclass
class ExtractedPage:
    """One page worth of text + table content."""
    page: int
    text: str
    tables: list[ExtractedTable] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip()) or any(t.is_valid() for t in self.tables)


def _clean_cell(cell: Any) -> str:
    """Normalize a raw cell value to a clean string."""
    if cell is None:
        return ""
    s = str(cell)
    # Collapse whitespace and newlines that pdfplumber leaves in cells
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_header(row: list[str], next_row: list[str]) -> bool:
    """Heuristic: classify a row as a header if it's a label/year row, with
    actual measurement data below it.

    Year columns ("2022", "2023", "2024") in a header row are a STRONG positive
    signal — they're how comparison tables are typically laid out — so we
    cannot treat "this row has numbers" as a header-disqualifier.
    """
    if not row or not next_row:
        return False

    # Strong positive: multiple years in the row → almost certainly a header
    years_in_row = sum(
        1 for c in row if c and re.search(r"\b(19|20)\d{2}\b", c)
    )
    if years_in_row >= 2:
        return True

    # Otherwise: look for "non-year numbers" specifically
    def has_measurement_number(cell: str) -> bool:
        if not cell:
            return False
        # Strip year-like substrings, then check for what's left
        stripped = re.sub(r"\b(19|20)\d{2}\b", "", cell)
        # Measurements typically have either a decimal point, a thousands
        # separator, or a multi-digit run
        return bool(re.search(r"\d[\d,.\s]*\.\d|\d{3,}", stripped))

    row_has_data = any(has_measurement_number(c) for c in row if c)
    next_has_data = any(has_measurement_number(c) for c in next_row if c)
    return not row_has_data and next_has_data


def _drop_caption_rows(rows: list[list[str]]) -> list[list[str]]:
    """Drop leading rows that look like captions/titles, not data.

    A caption row has exactly one non-empty cell (e.g., a chart title that
    pdfplumber swept into the table extraction). Stop dropping as soon as
    we hit a row with multiple populated cells.
    """
    cleaned = []
    started = False
    for row in rows:
        n_filled = sum(1 for c in row if c and c.strip())
        if not started and n_filled <= 1:
            # Still in the caption zone — skip
            continue
        started = True
        cleaned.append(row)
    return cleaned


def _extract_tables_from_page(page, page_num: int) -> list[ExtractedTable]:
    """Run table extraction with primary settings; fall back if nothing found."""
    tables = []
    table_objects = page.find_tables(table_settings=DEFAULT_TABLE_SETTINGS)

    # If lines-based detection finds nothing, try text-based as a fallback
    if not table_objects:
        try:
            table_objects = page.find_tables(table_settings=FALLBACK_TABLE_SETTINGS)
        except Exception:
            table_objects = []

    for idx, t in enumerate(table_objects):
        try:
            raw_rows = t.extract()
        except Exception as e:
            logger.debug(f"  table extract failed on page {page_num}: {e}")
            continue

        # Normalize cells, drop fully empty rows, then strip caption rows
        rows = [[_clean_cell(c) for c in row] for row in raw_rows or []]
        rows = [r for r in rows if any(c for c in r)]
        rows = _drop_caption_rows(rows)
        if not rows:
            continue

        # Detect header: try row 0 against row 1
        header = (rows[0] if len(rows) >= 2 and _looks_like_header(rows[0], rows[1])
                  else None)

        tbl = ExtractedTable(
            table_id=f"p{page_num}_t{idx}",
            page=page_num,
            bbox=t.bbox,
            rows=rows,
            header_row=header,
        )
        if tbl.is_valid():
            tables.append(tbl)

    return tables


def _extract_text_excluding_tables(page, tables: list[ExtractedTable]) -> str:
    """Get page text, then mask out regions covered by tables.

    This prevents double-counting: numbers in a table shouldn't also appear
    as scattered text in the narrative stream (which is what creates the
    'embedded numbers without context' problem the audit flagged).
    """
    if not tables:
        # No tables → just get all text
        text = page.extract_text() or ""
    else:
        # Crop the page to exclude table regions
        bboxes = [t.bbox for t in tables]
        # pdfplumber doesn't have a built-in "extract text outside these bboxes",
        # so we filter chars manually
        chars_outside = [
            c for c in page.chars
            if not any(_char_in_bbox(c, bb) for bb in bboxes)
        ]
        if chars_outside:
            text = "".join(c["text"] for c in chars_outside)
            # Re-add line breaks by detecting y-coordinate jumps
            text = _reconstruct_lines(chars_outside)
        else:
            text = ""

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _char_in_bbox(char: dict, bbox: tuple) -> bool:
    """Is the char inside the table bounding box?"""
    x0, y0, x1, y1 = bbox
    cx = (char["x0"] + char["x1"]) / 2
    cy = (char["top"] + char["bottom"]) / 2
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _reconstruct_lines(chars: list[dict]) -> str:
    """Rebuild line-broken text from a sorted char stream."""
    if not chars:
        return ""
    chars_sorted = sorted(chars, key=lambda c: (round(c["top"]), c["x0"]))
    lines, current_line, current_top = [], [], None
    for c in chars_sorted:
        top = round(c["top"])
        if current_top is None or abs(top - current_top) > 3:
            if current_line:
                lines.append("".join(current_line))
            current_line = [c["text"]]
            current_top = top
        else:
            current_line.append(c["text"])
    if current_line:
        lines.append("".join(current_line))
    return "\n".join(lines)


def parse_pdf(
    pdf_path: Path,
    max_pages: int | None = None,
    narrative_only: bool = False,
) -> Iterator[ExtractedPage]:
    """Parse a PDF page by page, yielding ExtractedPage objects.

    Yields lazily so very large PDFs don't blow up memory.

    narrative_only=True bypasses table detection entirely and emits only
    page text. Use for prose-heavy docs where pdfplumber's "lines" detector
    misreads visual layout (e.g. the EUR-Lex Official Journal's two-column
    format) as table structure, shredding phrases across phantom cells.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        if max_pages:
            n_pages = min(n_pages, max_pages)

        for i in range(n_pages):
            page_num = i + 1
            page = pdf.pages[i]
            try:
                if narrative_only:
                    tables = []
                    # No tables → _extract_text_excluding_tables returns the
                    # full, whitespace-normalized page text (reuses existing
                    # logic instead of duplicating it).
                    text = _extract_text_excluding_tables(page, [])
                else:
                    tables = _extract_tables_from_page(page, page_num)
                    text = _extract_text_excluding_tables(page, tables)
            except Exception as e:
                logger.warning(f"  {pdf_path.name} p{page_num} parse error: {e}")
                continue

            extracted = ExtractedPage(page=page_num, text=text, tables=tables)
            if extracted.has_content:
                yield extracted


def parse_pdf_to_list(
    pdf_path: Path,
    max_pages: int | None = None,
    narrative_only: bool = False,
) -> list[ExtractedPage]:
    """Eager version — returns all pages as a list. Use for small docs."""
    return list(parse_pdf(pdf_path, max_pages=max_pages, narrative_only=narrative_only))


# ──────────────────────────────────────────────────────────────────────────
# Quick stats for diagnostics
# ──────────────────────────────────────────────────────────────────────────

def parse_stats(pdf_path: Path) -> dict:
    """Return per-doc stats: page count, table count, char count.
    Useful for quickly checking 'did table extraction find anything'."""
    n_pages, n_tables, n_chars = 0, 0, 0
    for p in parse_pdf(pdf_path):
        n_pages += 1
        n_tables += len(p.tables)
        n_chars += len(p.text)
        for t in p.tables:
            n_chars += sum(len(c) for row in t.rows for c in row)
    return {
        "file": str(pdf_path.name),
        "pages": n_pages,
        "tables_found": n_tables,
        "chars_extracted": n_chars,
    }


if __name__ == "__main__":
    # Smoke test
    import sys
    if len(sys.argv) != 2:
        print("Usage: python parser.py <pdf_path>")
        sys.exit(1)
    stats = parse_stats(Path(sys.argv[1]))
    print(stats)
