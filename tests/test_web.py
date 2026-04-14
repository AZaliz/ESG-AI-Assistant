from pathlib import Path

from app.rag import DEFAULT_SAMPLE_PDF, parse_pdf_path_lines
from app.web import create_app


def test_parse_pdf_path_lines_uses_sample_when_blank():
    paths = parse_pdf_path_lines("")
    assert paths == [DEFAULT_SAMPLE_PDF.resolve()]


def test_web_homepage_renders_successfully():
    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(
        app(
            {
                "PATH_INFO": "/",
                "REQUEST_METHOD": "GET",
                "wsgi.input": __import__("io").BytesIO(b""),
                "CONTENT_LENGTH": "0",
            },
            start_response,
        )
    ).decode("utf-8")

    assert captured["status"] == "200 OK"
    assert "ESG RAG Studio" in body
    assert "Build Index" in body
