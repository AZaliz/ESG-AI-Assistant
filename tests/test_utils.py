from app.utils import normalize_url


def test_normalize_url_strips_tracking_params_and_www():
    url = normalize_url("https://www.example.com/report.pdf?utm_source=x&id=1")
    assert url == "https://example.com/report.pdf?id=1"


def test_normalize_url_resolves_relative_paths():
    url = normalize_url("../reports/latest.pdf", "https://example.com/sustainability/2025/index.html")
    assert url == "https://example.com/sustainability/reports/latest.pdf"
