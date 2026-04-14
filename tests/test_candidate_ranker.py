from app.candidate_ranker import rank_candidates
from app.models import CandidateLink, CompanyProfile, CompanySeed


def test_rank_candidates_prefers_recent_pdf_sustainability_report():
    profile = CompanyProfile(
        seed=CompanySeed(name="ASML", ranking_source="manual", ranking_url="manual"),
        canonical_name="ASML",
        issuer_domains=["asml.com"],
        source_urls=[],
    )
    candidates = [
        CandidateLink(
            company_name="ASML",
            url="https://www.asml.com/en/investors/reports/asml-sustainability-report-2025.pdf",
            source_page_url="https://www.asml.com/en/investors",
            source_type="issuer_pdf",
            anchor_text="Sustainability Report 2025",
            page_title="Investor relations",
            nearby_text="Latest sustainability report",
            file_name="asml-sustainability-report-2025.pdf",
        ),
        CandidateLink(
            company_name="ASML",
            url="https://www.asml.com/en/news/press-release-q1-2026",
            source_page_url="https://www.asml.com/en/news",
            source_type="issuer_html",
            anchor_text="Press release Q1 2026",
            page_title="News",
            nearby_text="Quarterly news update",
            file_name="press-release-q1-2026",
        ),
    ]
    ranked = rank_candidates(candidates, profile)
    assert ranked[0].url.endswith(".pdf")
    assert ranked[0].score > ranked[1].score

