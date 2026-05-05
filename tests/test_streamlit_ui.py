from app.streamlit_ui import _build_document_prompt


def test_build_document_prompt_includes_uploaded_document_context():
    document = {
        "name": "report.pdf",
        "suffix": ".pdf",
        "page_count": 12,
        "summary": "Document uploaded successfully.",
        "text": "Scope 1 emissions are reduced by 40% by 2030.",
    }

    prompt = _build_document_prompt("What are the targets?", document)

    assert "report.pdf" in prompt
    assert "Scope 1 emissions" in prompt
    assert "What are the targets?" in prompt


def test_build_document_prompt_truncates_very_long_documents():
    document = {
        "name": "long.txt",
        "suffix": ".txt",
        "page_count": 0,
        "summary": "Document uploaded successfully.",
        "text": "x" * 25000,
    }

    prompt = _build_document_prompt("Summarize it.", document)

    assert "[Document truncated for context length]" in prompt