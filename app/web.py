"""Lightweight browser UI for the ESG RAG workflow."""

from __future__ import annotations

import html
import os
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from app.rag import (
    DEFAULT_INDEX_DIR,
    DEFAULT_SAMPLE_PDF,
    DEFAULT_BASE_URL,
    answer_question,
    build_index,
    get_index_summary,
    parse_pdf_path_lines,
    retrieve_chunks,
    set_api_key,
)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _render_page(
    *,
    index_dir: Path,
    build_result: dict[str, Any] | None = None,
    ask_result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> str:
    summary = get_index_summary(index_dir)
    summary_html = _render_index_summary(summary, index_dir)
    build_html = _render_build_result(build_result)
    ask_html = _render_ask_result(ask_result)
    error_html = (
        f'<section class="panel panel-error"><h2>Issue</h2><pre>{_escape(error_message)}</pre></section>'
        if error_message
        else ""
    )
    sample_value = _escape(str(DEFAULT_SAMPLE_PDF.resolve()))
    current_key_hint = "present" if os.environ.get("ALBERT_API_KEY") else "missing"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESG AI Terminal</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-deep: #07080e;
      --bg-surface: #0d0f1a;
      --bg-elevated: #131522;
      --bg-input: #0a0b14;
      --border: #1a1c2e;
      --border-hover: #252840;
      --text-primary: #e1e4f0;
      --text-secondary: #8b8fa8;
      --text-muted: #5c607a;
      --accent-green: #10b981;
      --accent-green-strong: #059669;
      --accent-coral: #f43f5e;
      --accent-cyan: #22d3ee;
      --glow-green: rgba(16,185,129,0.18);
      --glow-coral: rgba(244,63,94,0.18);
      --radius: 10px;
      --radius-sm: 6px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "JetBrains Mono", "IBM Plex Mono", "SF Mono", "Menlo", "Cascadia Code", monospace;
      font-size: 13px;
      line-height: 1.65;
      color: var(--text-primary);
      background: var(--bg-deep);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(16,185,129,0.04), transparent),
        radial-gradient(ellipse 60% 40% at 80% 85%, rgba(34,211,238,0.03), transparent);
      pointer-events: none;
      z-index: 0;
    }}
    .shell {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 22px 52px;
      position: relative;
      z-index: 1;
    }}
    .hero {{
      padding: 24px 26px;
      border: 1px solid var(--border);
      background: var(--bg-surface);
      border-radius: var(--radius);
      margin-bottom: 20px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
      opacity: 0.5;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 1.5rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text-primary);
    }}
    .hero h1 span {{
      color: var(--accent-green);
    }}
    .hero p {{
      margin: 0;
      max-width: 80ch;
      color: var(--text-secondary);
      font-size: 0.8rem;
    }}
    .hero-badges {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .badge {{
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 5px 10px;
      font-size: 0.7rem;
      background: var(--bg-elevated);
      color: var(--text-secondary);
      letter-spacing: 0.02em;
    }}
    .badge.ok {{ border-color: rgba(16,185,129,0.25); color: var(--accent-green); }}
    .badge.warn {{ border-color: rgba(244,63,94,0.25); color: var(--accent-coral); }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr 1fr;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
    }}
    .panel h2 {{
      margin: 0 0 8px;
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .panel h2::before {{
      content: ">";
      color: var(--accent-green);
      font-weight: 400;
    }}
    .panel p, .panel li, .meta {{
      color: var(--text-secondary);
      font-size: 0.75rem;
      margin: 0 0 10px;
    }}
    .panel-error {{
      border-color: rgba(244,63,94,0.3);
      background: rgba(244,63,94,0.06);
    }}
    .panel-error h2::before {{
      color: var(--accent-coral);
    }}
    form {{
      display: grid;
      gap: 12px;
    }}
    .row {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    label {{
      display: grid;
      gap: 5px;
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    input, textarea {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      font-family: inherit;
      font-size: 0.78rem;
      color: var(--text-primary);
      background: var(--bg-input);
      outline: none;
      transition: border-color 160ms ease;
    }}
    input:focus, textarea:focus {{
      border-color: var(--accent-green);
      box-shadow: 0 0 0 2px var(--glow-green);
    }}
    input:read-only {{
      color: var(--text-muted);
      cursor: default;
    }}
    textarea {{
      min-height: 100px;
      resize: vertical;
      line-height: 1.5;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--accent-green);
      border-radius: var(--radius-sm);
      background: rgba(16,185,129,0.1);
      color: var(--accent-green);
      padding: 8px 16px;
      font-family: inherit;
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 140ms ease, box-shadow 140ms ease;
      letter-spacing: 0.02em;
    }}
    button:hover {{
      background: rgba(16,185,129,0.18);
      box-shadow: 0 0 12px var(--glow-green);
    }}
    .muted {{
      color: var(--text-muted);
      font-size: 0.68rem;
    }}
    .stats {{
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 10px;
    }}
    .stat {{
      border: 1px solid var(--border);
      background: var(--bg-elevated);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--accent-green);
      margin-bottom: 2px;
    }}
    .stat span {{
      color: var(--text-muted);
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px;
      margin: 0;
      font-family: inherit;
      font-size: 0.78rem;
      line-height: 1.55;
      color: var(--text-secondary);
      overflow-x: auto;
    }}
    .chunk {{
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      border-radius: var(--radius-sm);
      background: var(--bg-elevated);
      border: 1px solid var(--border);
    }}
    .chunk + .chunk {{
      margin-top: 10px;
    }}
    .chunk-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px 10px;
      font-size: 0.68rem;
      color: var(--text-muted);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      background: rgba(16,185,129,0.1);
      color: var(--accent-green);
      font-size: 0.65rem;
      border: 1px solid rgba(16,185,129,0.15);
    }}
    ul {{
      padding-left: 16px;
      color: var(--text-secondary);
      font-size: 0.75rem;
    }}
    @media (max-width: 840px) {{
      .grid, .row, .stats {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        padding: 16px 10px 36px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>ESG_AI <span>v1.0</span></h1>
      <p>Build a searchable ESG knowledge base from PDF reports — index → embed → retrieve → answer.</p>
      <div class="hero-badges">
        <span class="badge {('ok' if current_key_hint == 'present' else 'warn')}">API key: {_escape(current_key_hint)}</span>
        <span class="badge">&gt; index {_escape(str(index_dir))}</span>
      </div>
    </section>

    {error_html}

    <section class="grid">
      <div class="stack">
        <section class="panel">
          <h2>Build Index</h2>
          <p>Paste absolute PDF paths (one per line). Leave the sample path to rebuild the demo index.</p>
          <form method="post" action="/build">
            <label>
              Albert API Key
              <input type="password" name="api_key" placeholder="sk-... or leave blank for env var">
            </label>
            <label>
              PDF paths
              <textarea name="pdf_paths">{sample_value}</textarea>
            </label>
            <div class="row">
              <label>
                Chunk target tokens
                <input type="number" name="chunk_target_tokens" value="420" min="100" max="1000">
              </label>
              <label>
                Batch size
                <input type="number" name="batch_size" value="16" min="1" max="128">
              </label>
            </div>
            <div class="actions">
              <button type="submit">build</button>
              <span class="muted">FAISS auto-detected; NumPy fallback available.</span>
            </div>
          </form>
        </section>

        <section class="panel">
          <h2>Ask Question</h2>
          <p>Query the current index, inspect retrieval, and generate a grounded answer.</p>
          <form method="post" action="/ask">
            <label>
              Albert API Key
              <input type="password" name="api_key" placeholder="sk-... or leave blank for env var">
            </label>
            <label>
              ESG question
              <textarea name="question">What climate-related targets are disclosed for 2030, and are they science-based?</textarea>
            </label>
            <div class="row">
              <label>
                Top K chunks
                <input type="number" name="top_k" value="5" min="1" max="20">
              </label>
              <label>
                Mode
                <input type="text" value="Grounded answer + retrieval" readonly>
              </label>
            </div>
            <div class="actions">
              <button type="submit">query</button>
            </div>
          </form>
        </section>
      </div>

      <div class="stack">
        {summary_html}
        {build_html}
        {ask_html}
      </div>
    </section>
  </main>
</body>
</html>
"""


def _render_index_summary(summary: dict[str, Any] | None, index_dir: Path) -> str:
    if not summary:
        return f"""
<section class="panel">
  <h2>Index Status</h2>
  <p>No index built yet in <strong>{_escape(str(index_dir))}</strong>. Run <span style="color:var(--accent-green)">build</span> from the left panel to start querying.</p>
</section>
"""

    pdf_list = "".join(f"<li>{_escape(pdf)}</li>" for pdf in summary.get("pdfs", []))
    return f"""
<section class="panel">
  <h2>Index Status</h2>
  <div class="stats">
    <div class="stat"><strong>{_escape(summary.get("chunk_count", 0))}</strong><span>Chunks</span></div>
    <div class="stat"><strong>{_escape(summary.get("vector_backend", "unknown"))}</strong><span>Vector backend</span></div>
    <div class="stat"><strong>{_escape(summary.get("embedding_model", "not built"))}</strong><span>Embedding model</span></div>
    <div class="stat"><strong>{_escape(summary.get("embedding_dimension", "n/a"))}</strong><span>Dimensions</span></div>
  </div>
  <p class="meta">Built: {_escape(summary.get("built_at", "unknown"))}</p>
  <ul>{pdf_list}</ul>
</section>
"""


def _render_build_result(build_result: dict[str, Any] | None) -> str:
    if not build_result:
        return ""
    return f"""
<section class="panel">
  <h2>Latest Build</h2>
  <pre>{_escape(json_dumps_pretty(build_result))}</pre>
</section>
"""


def _render_ask_result(ask_result: dict[str, Any] | None) -> str:
    if not ask_result:
        return ""

    chunks_html = "".join(
        f"""
<div class="chunk">
  <div class="chunk-header">
    <span class="pill">{_escape(chunk["chunk_id"])}</span>
    <span>{_escape(chunk["source_file"])}</span>
    <span>pages {_escape(chunk["page_start"])}-{_escape(chunk["page_end"])}</span>
    <span>score {_escape(f'{chunk["score"]:.4f}')}</span>
  </div>
  <pre>{_escape(chunk["text"])}</pre>
</div>
"""
        for chunk in ask_result["retrieved_chunks"]
    )
    return f"""
<section class="panel">
  <h2>Grounded Answer</h2>
  <p class="meta">Embedding model: {_escape(ask_result["embedding_model"])} | Text model: {_escape(ask_result["text_model"])}</p>
  <pre>{_escape(ask_result["answer"])}</pre>
</section>
<section class="panel">
  <h2>Retrieved Chunks</h2>
  {chunks_html}
</section>
"""


def json_dumps_pretty(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)


def _read_form(environ: dict[str, Any]) -> dict[str, str]:
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
    except ValueError:
        length = 0
    body = environ["wsgi.input"].read(length).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def _build_handler(form: dict[str, str], index_dir: Path) -> dict[str, Any]:
    set_api_key(form.get("api_key"))
    pdf_paths = parse_pdf_path_lines(form.get("pdf_paths", ""))
    return build_index(
        pdf_paths=pdf_paths,
        index_dir=index_dir,
        target_tokens=int(form.get("chunk_target_tokens", "420") or 420),
        min_tokens=300,
        max_tokens=500,
        batch_size=int(form.get("batch_size", "16") or 16),
        base_url=DEFAULT_BASE_URL,
    )


def _ask_handler(form: dict[str, str], index_dir: Path) -> dict[str, Any]:
    set_api_key(form.get("api_key"))
    question = form.get("question", "").strip()
    if not question:
        raise RuntimeError("Please enter a question before submitting.")

    retrieved_chunks, embedding_model = retrieve_chunks(
        index_dir=index_dir,
        question=question,
        top_k=int(form.get("top_k", "5") or 5),
        base_url=DEFAULT_BASE_URL,
    )
    answer, text_model = answer_question(
        question=question,
        retrieved_chunks=retrieved_chunks,
        base_url=DEFAULT_BASE_URL,
    )
    return {
        "question": question,
        "embedding_model": embedding_model,
        "text_model": text_model,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }


def create_app(index_dir: Path):
    def app(environ: dict[str, Any], start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()
        status = "200 OK"
        build_result: dict[str, Any] | None = None
        ask_result: dict[str, Any] | None = None
        error_message: str | None = None

        try:
            if path == "/build" and method == "POST":
                form = _read_form(environ)
                build_result = _build_handler(form, index_dir)
            elif path == "/ask" and method == "POST":
                form = _read_form(environ)
                ask_result = _ask_handler(form, index_dir)
            elif path != "/":
                status = "404 Not Found"
                error_message = f"Unknown route: {path}"
        except Exception as exc:  # noqa: BLE001
            error_message = f"{exc}\n\n{traceback.format_exc(limit=2)}"
            status = "500 Internal Server Error"

        body = _render_page(
            index_dir=index_dir,
            build_result=build_result,
            ask_result=ask_result,
            error_message=error_message,
        ).encode("utf-8")
        headers = [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))]
        start_response(status, headers)
        return [body]

    return app


def run_web_app(host: str, port: int, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    app = create_app(index_dir=index_dir)
    print(f"Serving ESG RAG Studio on http://{host}:{port}")
    with make_server(host, port, app) as server:
        server.serve_forever()
