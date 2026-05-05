from pathlib import Path

from app.streamlit_ui import _build_document_prompt, _chunk_document_text, _select_local_report


def test_chunk_document_text_uses_overlap():
    text = "one two three four five six seven eight"

    chunks = _chunk_document_text(text, chunk_size=4, chunk_overlap=2)

    assert chunks == [
        "one two three four",
        "three four five six",
        "five six seven eight",
    ]


def test_build_document_prompt_includes_uploaded_document_context_and_chunk_settings():
    document = {
        "name": "report.pdf",
        "suffix": ".pdf",
        "page_count": 12,
        "summary": "Document uploaded successfully.",
        "text": "Scope 1 emissions are reduced by 40% by 2030. Net-zero by 2050.",
    }

    prompt = _build_document_prompt("What are the targets?", document, chunk_size=4, chunk_overlap=2)

    assert "report.pdf" in prompt
    assert "Scope 1 emissions" in prompt
    assert "What are the targets?" in prompt
    assert "Chunk size: 4 words" in prompt
    assert "Chunk overlap: 2 words" in prompt
    assert "Chunk 1/5" in prompt


def test_select_local_report_prefers_company_name_in_prompt():
    reports = [
        Path("sample_data/totalenergies_sustainability-climate-2025-progress-report_2025_en.pdf"),
        Path("sample_data/bnp_paribas_integrated_report_2024_en_bd_1.pdf"),
        Path("sample_data/airbus_report.pdf"),
    ]

    selected_path, score = _select_local_report("What are BNP Paribas ESG priorities?", "", reports)

    assert selected_path == reports[1]
    assert score > 55