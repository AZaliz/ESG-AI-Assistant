"""Shared base scraper: HTTP, download, dedup, file naming."""
from __future__ import annotations
import hashlib
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .models import Document, ESG_REPORT_DOC_TYPES

logger = logging.getLogger(__name__)

# Realistic browser UA — corporate WAFs (TotalEnergies, BNP, etc.) block
# anything that looks bot-like. We're polite via throttling instead.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 60
THROTTLE_SECONDS = 1.5  # between requests


class BaseScraper:
    """Subclass and implement discover()."""

    company: str = ""        # set in subclass
    hub_urls: list = []      # set in subclass

    def __init__(self, output_dir: Path):
        if not self.company:
            raise ValueError("Subclass must set 'company' attribute")
        self.output_dir = Path(output_dir) / self._slug(self.company)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # Full Chrome-like header set — basic Cloudflare/Akamai bot rules
        # check for the Sec-Fetch-* family in addition to User-Agent.
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            # br (brotli) is part of the "real Chrome" signature that BNP's
            # and Schneider's WAFs check for. Requires the `brotli` package
            # so urllib3 can decode the response.
            "Accept-Encoding": "gzip, deflate, br",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
        })
        self._warmed = False

    def _warm_session(self):
        """Visit the site root to pick up cookies before hitting target pages.
        Defeats most basic Cloudflare/Akamai challenges which look for a
        normal navigation history before allowing resource fetches."""
        if self._warmed or not self.hub_urls:
            return
        from urllib.parse import urlparse
        parsed = urlparse(self.hub_urls[0])
        root = f"{parsed.scheme}://{parsed.netloc}/"
        try:
            # First navigation: Sec-Fetch-Site: none (no referrer)
            r = self.session.get(
                root,
                headers={"Sec-Fetch-Site": "none"},
                timeout=DEFAULT_TIMEOUT,
            )
            logger.debug(
                f"  [{self.company}] warmed via {root} → {r.status_code} "
                f"({len(self.session.cookies)} cookies)"
            )
        except Exception as e:
            logger.debug(f"  [{self.company}] warmup failed (continuing): {e}")
        self._warmed = True

    # ── public interface ────────────────────────────────────────────────

    def discover(self) -> list:
        """Subclasses MUST implement this. Return list of Document objects.
        Should NOT actually download — that's done by run()."""
        raise NotImplementedError

    def run(self, dry_run: bool = False, allowed_doc_types: Optional[set] = None) -> list:
        """Discover then download. Returns list of successfully fetched docs.

        Documents whose doc_type is not in `allowed_doc_types` are dropped
        before download. Defaults to ESG_REPORT_DOC_TYPES (excludes policies).
        Pass `DOC_TYPES` to keep everything.
        """
        if allowed_doc_types is None:
            allowed_doc_types = ESG_REPORT_DOC_TYPES

        logger.info(f"=== {self.company} ===")
        self._warm_session()
        try:
            docs = self.discover()
        except Exception as e:
            logger.error(f"[{self.company}] discovery failed: {e}", exc_info=True)
            return []

        n_raw = len(docs)
        docs = [d for d in docs if d.doc_type in allowed_doc_types]
        n_dropped = n_raw - len(docs)
        if n_dropped:
            logger.info(f"  filtered {n_dropped} non-ESG-report doc(s) (e.g. policies)")

        logger.info(f"  discovered {len(docs)} documents")
        if dry_run:
            for d in docs:
                fy = d.fiscal_year or "----"
                logger.info(f"    [{fy}] {d.doc_type:23s} {d.title[:65]}")
            return docs

        ok = []
        for d in docs:
            result = self.download(d)
            if result:
                ok.append(result)
        logger.info(f"  downloaded {len(ok)}/{len(docs)}")
        return ok

    # ── helpers used by extractors ──────────────────────────────────────

    def fetch_html(self, url: str) -> BeautifulSoup:
        """Fetch a page, return parsed BeautifulSoup. Throttled."""
        time.sleep(THROTTLE_SECONDS)
        logger.debug(f"  GET {url}")
        r = self.session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")

    @staticmethod
    def extract_year(*texts: str) -> Optional[int]:
        """Find the most recent plausible 4-digit year in any of the inputs.
        Looks for 19xx or 20xx, capped at current year + 1."""
        all_years = []
        for t in texts:
            if not t:
                continue
            all_years.extend(int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", t))
        if not all_years:
            return None
        current = datetime.now().year
        valid = [y for y in all_years if 1995 <= y <= current + 1]
        return max(valid) if valid else None

    @staticmethod
    def file_format_from_url(url: str) -> str:
        """Infer file extension from URL (defaults to 'pdf')."""
        m = re.search(r"\.([a-z0-9]{2,5})(?:\?|#|$)", url.lower())
        return m.group(1) if m else "pdf"

    # ── download internals ──────────────────────────────────────────────

    def download(self, doc: Document) -> Optional[Document]:
        """Download a single document. Dedupes by content hash."""
        time.sleep(THROTTLE_SECONDS)
        try:
            r = self.session.get(doc.url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"  [fail] {doc.url}: {e}")
            return None

        content = r.content
        if len(content) < 1024:
            logger.warning(f"  [fail] {doc.url}: response too small ({len(content)} bytes)")
            return None

        digest = hashlib.sha256(content).hexdigest()
        path = self._build_path(doc, digest)

        if path.exists():
            logger.debug(f"  [skip] {path.name} already exists")
        else:
            path.write_bytes(content)
            logger.info(f"  [save] {path.name} ({len(content)/1e6:.1f} MB)")

        doc.file_path = path
        doc.file_hash = digest
        doc.size_bytes = len(content)
        doc.downloaded_at = datetime.now()
        return doc

    def _build_path(self, doc: Document, digest: str) -> Path:
        """Stable, sortable filename: {fiscal_year}_{doc_type}_{hash8}.{ext}"""
        fy = doc.fiscal_year or doc.publication_year or "undated"
        ext = doc.file_format or "pdf"
        return self.output_dir / f"{fy}_{doc.doc_type}_{digest[:8]}.{ext}"

    @staticmethod
    def _slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
