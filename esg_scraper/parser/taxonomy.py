"""Shared chunk-tagging taxonomy: speaker_role + esrs_topic.

Single source of truth so the PDF path (parser.chunker.build_chunks_for_document)
and the transcript path (ingest_earnings._speaker_chunks) tag chunks identically.
Imported by both, so a normal build reproduces these fields natively — no
post-hoc patch step.

  speaker_role : "executive" | "management" | "analyst" | "operator" | None
                 (None for every non-earnings_call chunk)
  esrs_topic   : sorted list, subset of {E1..E5, S1..S4, G1}; [] if no match.
                 Deliberately conservative — false negatives preferred over
                 false positives that pollute filters. Do NOT widen patterns
                 without an explicit decision.
"""
from __future__ import annotations

import re

# ── speaker_role ────────────────────────────────────────────────────────────
# Transcript turn headers carry one of these role tokens; map to the schema
# vocabulary. ("Attendees" = invited non-exec participants -> management.)
_ROLE_MAP = {
    "Executives": "executive",
    "Analysts": "analyst",
    "Operator": "operator",
    "Attendees": "management",
}


def speaker_role_for_turn(role_token: str | None) -> str | None:
    """Map a transcript turn's role token to the speaker_role vocabulary.

    Returns None for unknown/empty tokens (and is never called for
    non-earnings chunks, which keep the Chunk default of None).
    """
    if not role_token:
        return None
    return _ROLE_MAP.get(role_token)


# ── esrs_topic ──────────────────────────────────────────────────────────────
ESRS_PATTERNS = {
    "E1": re.compile(
        r"\b(scope\s+[123]|GHG\s+emissions?|greenhouse\s+gas|"
        r"climate\s+transition|net.?zero|carbon\s+neutral|"
        r"tCO2|MtCO2|kCO2|emission\s+reduction\s+target|"
        r"SBTi|science.based\s+target|carbon\s+budget|"
        r"renewable\s+energy|energy\s+consumption|fossil\s+fuel|"
        r"TCFD|climate.related\s+risk)\b",
        re.IGNORECASE,
    ),
    "E2": re.compile(
        r"\b(air\s+pollution|water\s+pollution|soil\s+contamination|"
        r"pollutant|substances?\s+of\s+(very\s+high\s+)?concern|"
        r"emissions?\s+to\s+(air|water|soil)|hazardous\s+substance)\b",
        re.IGNORECASE,
    ),
    "E3": re.compile(
        r"\b(water\s+(withdrawal|consumption|stress|scarcity|usage)|"
        r"marine\s+resources?|freshwater|water.stressed)\b",
        re.IGNORECASE,
    ),
    "E4": re.compile(
        r"\b(biodiversity|ecosystem|protected\s+area|species|"
        r"deforestation|habitat|land.use\s+change|natural\s+capital)\b",
        re.IGNORECASE,
    ),
    "E5": re.compile(
        r"\b(circular\s+economy|waste\s+(generated|recycl|recover|management)|"
        r"resource\s+(use|efficiency)|packaging|materials?\s+used|"
        r"renewable\s+material)\b",
        re.IGNORECASE,
    ),
    "S1": re.compile(
        r"\b(own\s+workforce|employee|headcount|gender\s+pay\s+gap|"
        r"women\s+in\s+(management|leadership)|training\s+hours?|"
        r"LTIFR|TRIR|occupational\s+(health|safety)|collective\s+bargaining|"
        r"living\s+wage|workforce\s+turnover)\b",
        re.IGNORECASE,
    ),
    "S2": re.compile(
        r"\b(value\s+chain\s+workers?|supply\s+chain\s+(audit|due\s+diligence)|"
        r"supplier\s+(audit|assessment)|forced\s+labou?r|child\s+labou?r|"
        r"modern\s+slavery|palm\s+oil|conflict\s+mineral)\b",
        re.IGNORECASE,
    ),
    "S3": re.compile(
        r"\b(affected\s+communit|indigenous|local\s+communit|"
        r"land\s+rights|community\s+investment|displacement|"
        r"free,?\s+prior\s+and\s+informed\s+consent|FPIC)\b",
        re.IGNORECASE,
    ),
    "S4": re.compile(
        r"\b(consumers?\s+and\s+end.users?|product\s+safety|"
        r"data\s+privacy|consumer\s+health|advertising\s+practice|"
        r"product\s+labelling|customer\s+complaint)\b",
        re.IGNORECASE,
    ),
    "G1": re.compile(
        r"\b(business\s+conduct|anti.corruption|bribery|whistleblower|"
        r"lobbying|political\s+contribution|payment\s+practice|"
        r"corporate\s+culture|code\s+of\s+(conduct|ethics))\b",
        re.IGNORECASE,
    ),
}


def classify_esrs_topics(
    text: str,
    chunk_kind: str | None = None,
    raw_label: str | None = None,
) -> list[str]:
    """Sorted ESRS topics matched in `text`.

    For short table_fact chunks (<30 chars of text) the sentence is too
    sparse, so `raw_label` is appended to the matched scope.
    """
    t = text or ""
    if chunk_kind == "table_fact" and len((t).strip()) < 30 and raw_label:
        t = t + " " + str(raw_label)
    return sorted(topic for topic, pat in ESRS_PATTERNS.items() if pat.search(t))
