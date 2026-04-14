"""SQLite-backed metadata storage and file-based candidate logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.models import CandidateLink, DocumentRecord
from app.utils import CANDIDATE_DIR, DB_PATH, OUTPUT_DIR, ensure_directories, slugify


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    company_name TEXT NOT NULL,
    ticker TEXT,
    country TEXT,
    report_year INTEGER,
    document_type TEXT,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    final_url TEXT NOT NULL PRIMARY KEY,
    source_type TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    download_path TEXT NOT NULL,
    text_path TEXT,
    file_hash TEXT NOT NULL,
    published_date TEXT,
    discovery_confidence REAL NOT NULL,
    parse_status TEXT NOT NULL,
    notes TEXT,
    parser_used TEXT,
    retrieved_at TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        ensure_directories()
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute(SCHEMA_SQL)
        self.connection.commit()

    def save_document(self, record: DocumentRecord) -> None:
        payload = record.model_dump()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO documents (
                company_name, ticker, country, report_year, document_type, title,
                source_url, final_url, source_type, mime_type, download_path, text_path,
                file_hash, published_date, discovery_confidence, parse_status, notes,
                parser_used, retrieved_at
            ) VALUES (
                :company_name, :ticker, :country, :report_year, :document_type, :title,
                :source_url, :final_url, :source_type, :mime_type, :download_path, :text_path,
                :file_hash, :published_date, :discovery_confidence, :parse_status, :notes,
                :parser_used, :retrieved_at
            )
            """,
            payload,
        )
        self.connection.commit()

    def save_candidates(self, company_name: str, candidates: list[CandidateLink]) -> Path:
        path = CANDIDATE_DIR / f"{slugify(company_name)}.json"
        data = [candidate.model_dump() for candidate in candidates]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_smoke_outputs(self, rows: list[dict], stem: str) -> tuple[Path, Path]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = OUTPUT_DIR / f"{stem}.json"
        csv_path = OUTPUT_DIR / f"{stem}.csv"
        json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        if rows:
            headers = list(rows[0].keys())
        else:
            headers = []
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(_csv_escape(str(row.get(header, ""))) for header in headers))
        csv_path.write_text("\n".join(lines), encoding="utf-8")
        return json_path, csv_path


def _csv_escape(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'
