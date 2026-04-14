"""Live smoke test for the current top European companies by market cap."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from urllib.parse import urlparse

from app.models import SmokeTestRow
from app.pipeline import AcquisitionPipeline
from app.ranking_sources import CompaniesMarketCapAdapter
from app.storage import Storage


LOGGER = logging.getLogger(__name__)


def run_smoke_test(top_n: int = 10) -> tuple[list[SmokeTestRow], str, str]:
    storage = Storage()
    pipeline = AcquisitionPipeline(storage=storage)
    ranking = CompaniesMarketCapAdapter(pipeline.session)
    seeds = ranking.get_top_european_companies(limit=top_n)

    rows: list[SmokeTestRow] = []
    for seed in seeds:
        LOGGER.info("Smoke testing company: %s", seed.name)
        profile, record, notes = pipeline.fetch_best(seed)
        resolved_domain = profile.issuer_domains[0] if profile.issuer_domains else None
        if not resolved_domain and record:
            host = urlparse(record.final_url).netloc.lower()
            if host and "companiesmarketcap.com" not in host:
                resolved_domain = host
        rows.append(
            SmokeTestRow(
                company=seed.name,
                discovered_issuer_domain=resolved_domain,
                chosen_report_title=record.title if record else None,
                report_year=record.report_year if record else None,
                source_url=record.final_url if record else None,
                file_type=record.mime_type if record else None,
                confidence=record.discovery_confidence if record else None,
                parse_status=record.parse_status if record else None,
                notes=" | ".join(notes),
            )
        )

    stem = f"smoke_test_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    row_dicts = [row.model_dump() for row in rows]
    json_path, csv_path = storage.save_smoke_outputs(row_dicts, stem=stem)
    return rows, str(json_path), str(csv_path)
