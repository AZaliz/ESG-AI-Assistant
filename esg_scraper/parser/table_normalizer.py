"""Table normalizer: convert ExtractedTable rows to natural-language sentences.

The core RAG problem with raw tables: a chunk like "Scope 1 | 2024 | 38.2 | tCO2e"
embeds poorly because it has no semantic context. The cell next to "38.2" might
be "2024" or "MtCO2e" but the embedding model doesn't know the column meanings.

The fix is to detect each table's schema (header column + value columns) and
emit one sentence per (header_cell, value_cell) pair, like:
    "Total Scope 1 GHG emissions for 2024 are 38.2 MtCO2e (TotalEnergies)."

This is what the audit said was needed:
  > "Adding table-aware extraction (e.g. Camelot, pdfplumber's extract_tables,
  > or layout-aware models) is the single highest-impact improvement."

Three table shapes we handle:
  1. KPI × years (most common in ESRS reports): rows are KPIs, columns are years
  2. KPI × Scope/method (Scope 1, Scope 2 LB, Scope 2 MB, etc.)
  3. Long lists (Article 8 Taxonomy): rows are activities, columns are alignment %s

For shapes we don't recognize, we emit row-by-row "linearized" sentences.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .parser import ExtractedTable

logger = logging.getLogger(__name__)


# Patterns to detect what each column header is about
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
UNIT_PATTERN = re.compile(
    r"\b(tCO2e?|tCO2eq|ktCO2e?|MtCO2e?|MWh|GWh|TWh|GJ|m3|m³|"
    r"hectares?|ha|kt|Mt|tonnes?|tons?|kWh|gCO2|"
    r"%|percent|EUR|€|USD|\$|persons?|FTE|employees?)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"-?\b\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?\b")

# Leading reference-code columns that hide the real KPI name one column over
# (e.g. Airbus ESG Datasheet's "Disclosure ref" = "ESRS E1-6").
REF_CODE_PATTERN = re.compile(
    r"^(ESRS\s+[ESG]\d+-\d+|GRI\s+\d+|SASB|TCFD)\b", re.IGNORECASE
)
# A cell that is essentially just a number (value, not a label).
PURE_NUMERIC_PATTERN = re.compile(r"^[-+]?[\d.,\s%()]+$")
COLUMN_NOISE_PATTERN = re.compile(r"^column\s*\d+$", re.IGNORECASE)


def _is_numeric_cell(cell: str) -> bool:
    s = (cell or "").strip()
    return bool(s) and bool(PURE_NUMERIC_PATTERN.match(s)) and any(ch.isdigit() for ch in s)


def make_label(row: list[str]) -> str | None:
    """Pick an informative row label, or None to signal 'drop this chunk'.

    Heuristic A — if row[0] is a reference code, junk (<=3 chars), or
        empty/whitespace, fall back to row[1].
    Heuristic B — if row[0] IS a reference code AND row[1] is non-numeric,
        keep both: "CODE | KPI name" (preserves traceability + searchable).
    Guard — final label < 5 chars or matching "column N" => None (noise).
    """
    if not row:
        return None
    first = (row[0] or "").strip()
    second = (row[1].strip() if len(row) > 1 and row[1] else "")

    is_code = bool(REF_CODE_PATTERN.match(first))
    trigger_a = is_code or len(first) <= 3 or first == ""

    if trigger_a:
        if is_code and second and not _is_numeric_cell(second):
            label = f"{first} | {second}"          # Heuristic B
        else:
            label = second                          # Heuristic A
    else:
        label = first

    label = label.strip()
    if len(label) < 5 or COLUMN_NOISE_PATTERN.match(label):
        return None
    return label


@dataclass
class NormalizedRow:
    """A single fact extracted from a table row, ready to embed."""
    sentence: str
    page: int
    table_id: str
    raw_label: str           # the KPI/row-header text
    raw_value: str           # the value as it appears in the cell
    raw_year: str | None = None
    raw_unit: str | None = None


def _detect_column_types(rows: list[list[str]]) -> dict[int, str]:
    """Classify each column as 'label', 'year', 'unit', 'value', or 'other'.

    Heuristic:
      - First column: usually 'label' (KPI name)
      - Columns with mostly years in their cells: 'year'
      - Columns with mostly unit strings: 'unit'
      - Columns with mostly numbers: 'value'
      - Everything else: 'other'
    """
    if not rows:
        return {}

    n_cols = max(len(r) for r in rows)
    col_types: dict[int, str] = {}

    for col_idx in range(n_cols):
        cells = [r[col_idx] for r in rows if col_idx < len(r)]
        non_empty = [c for c in cells if c and c.strip()]
        if not non_empty:
            col_types[col_idx] = "empty"
            continue

        year_hits = sum(1 for c in non_empty if YEAR_PATTERN.search(c))
        unit_hits = sum(1 for c in non_empty if UNIT_PATTERN.search(c) and not NUMBER_PATTERN.search(c))
        number_hits = sum(1 for c in non_empty if NUMBER_PATTERN.search(c))

        ratio = lambda x: x / len(non_empty)

        if col_idx == 0:
            col_types[col_idx] = "label"
        elif ratio(year_hits) > 0.5 and ratio(number_hits) < 0.7:
            col_types[col_idx] = "year"
        elif ratio(unit_hits) > 0.5:
            col_types[col_idx] = "unit"
        elif ratio(number_hits) > 0.4:
            col_types[col_idx] = "value"
        else:
            col_types[col_idx] = "other"

    return col_types


def _years_from_header(header_row: list[str]) -> dict[int, str]:
    """Pull years out of header cells: {col_idx: '2024'} for each col that names a year."""
    result = {}
    if not header_row:
        return result
    for i, cell in enumerate(header_row):
        if not cell:
            continue
        m = YEAR_PATTERN.search(cell)
        if m:
            result[i] = m.group(0)
    return result


def _is_kpi_by_year_table(table: ExtractedTable) -> bool:
    """Shape #1: rows = KPIs, columns = years. Most common in ESRS reports."""
    if not table.header_row:
        return False
    year_cols = _years_from_header(table.header_row)
    # At least 2 year columns means this is a year-comparison table
    return len(year_cols) >= 2


def _normalize_kpi_by_year(
    table: ExtractedTable,
    doc_meta: dict,
) -> list[NormalizedRow]:
    """Emit one sentence per (KPI, year) pair."""
    sentences = []
    year_cols = _years_from_header(table.header_row or [])
    if not year_cols:
        return []

    company = doc_meta.get("company", "the company")
    doc_type = doc_meta.get("doc_type", "report")

    # Skip header row when iterating data rows
    data_rows = table.rows[1:] if table.header_row else table.rows

    for row in data_rows:
        if not row:
            continue
        kpi_label = make_label(row)
        if kpi_label is None:  # uninformative/noise row -> drop the chunk
            continue

        # Extract unit if present in the label (e.g., "Scope 1 (MtCO2e)")
        unit_match = UNIT_PATTERN.search(kpi_label)
        unit = unit_match.group(0) if unit_match else None

        for col_idx, year in year_cols.items():
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if not cell or not cell.strip():
                continue
            # Cell should contain a number to be a fact
            if not NUMBER_PATTERN.search(cell):
                continue

            unit_str = f" {unit}" if unit and unit not in cell else ""
            sentence = (
                f"For {company}'s {doc_type}, {kpi_label} in {year} "
                f"is {cell.strip()}{unit_str}."
            )
            sentences.append(NormalizedRow(
                sentence=sentence,
                page=table.page,
                table_id=table.table_id,
                raw_label=kpi_label,
                raw_value=cell.strip(),
                raw_year=year,
                raw_unit=unit,
            ))

    return sentences


def _normalize_generic(
    table: ExtractedTable,
    doc_meta: dict,
) -> list[NormalizedRow]:
    """Fallback: linearize each row as 'Header: value, Header: value, ...'.

    Used when the table doesn't fit the KPI-by-year shape — e.g., Article 8
    Taxonomy tables where rows are economic activities and columns are
    alignment metrics across six environmental objectives.
    """
    sentences = []
    company = doc_meta.get("company", "the company")
    headers = table.header_row

    data_rows = table.rows[1:] if headers else table.rows

    for row in data_rows:
        if not row:
            continue
        label = make_label(row)
        if label is None:  # uninformative/noise row -> drop the chunk
            continue

        # Build "Header A: value A, Header B: value B" style
        pairs = []
        for i, cell in enumerate(row[1:], start=1):
            if not cell or not cell.strip():
                continue
            header_name = headers[i].strip() if headers and i < len(headers) else f"column {i}"
            if header_name:
                pairs.append(f"{header_name}: {cell.strip()}")

        if not pairs:
            continue

        sentence = (
            f"In {company}'s table on page {table.page}, "
            f"row '{label}' has {', '.join(pairs)}."
        )
        sentences.append(NormalizedRow(
            sentence=sentence,
            page=table.page,
            table_id=table.table_id,
            raw_label=label,
            raw_value=" | ".join(pairs),
        ))

    return sentences


def normalize_table(
    table: ExtractedTable,
    doc_meta: dict,
) -> list[NormalizedRow]:
    """Main entry: pick the right normalizer for the table shape.

    doc_meta should include at least: company, doc_type, fiscal_year.
    """
    if not table.is_valid():
        return []

    if _is_kpi_by_year_table(table):
        return _normalize_kpi_by_year(table, doc_meta)
    return _normalize_generic(table, doc_meta)


def normalize_all_tables(
    pages: list,
    doc_meta: dict,
) -> list[NormalizedRow]:
    """Normalize every table across every page of a parsed document."""
    all_rows = []
    for page in pages:
        for table in page.tables:
            all_rows.extend(normalize_table(table, doc_meta))
    return all_rows


# ──────────────────────────────────────────────────────────────────────────
# Sanity check helpers
# ──────────────────────────────────────────────────────────────────────────

def summarize_normalization(rows: list[NormalizedRow], top_n: int = 5) -> str:
    """Diagnostic: show what was extracted from the tables."""
    if not rows:
        return "  (no normalized rows produced)"

    lines = [f"  → produced {len(rows)} normalized sentences"]
    lines.append(f"  → first {top_n} examples:")
    for r in rows[:top_n]:
        lines.append(f"    [p.{r.page}] {r.sentence[:110]}")
    return "\n".join(lines)
