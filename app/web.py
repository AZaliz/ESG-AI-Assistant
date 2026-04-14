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
  <title>ESG RAG Studio</title>
  <style>
    :root {{
      --ink: #12263a;
      --ink-soft: #43556a;
      --paper: #f5f1e8;
      --card: #fffdf9;
      --line: #d5c8b3;
      --accent: #0f766e;
      --accent-strong: #0b5d58;
      --alert: #9f1239;
      --shadow: 0 20px 45px rgba(18, 38, 58, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 26%),
        linear-gradient(160deg, #efe6d5 0%, #f6f2ea 45%, #f0ecdf 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      padding: 28px 28px 22px;
      border: 1px solid rgba(18,38,58,0.12);
      background: linear-gradient(135deg, rgba(255,253,249,0.95), rgba(248,244,236,0.92));
      box-shadow: var(--shadow);
      border-radius: 24px;
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 5vw, 3.8rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      max-width: 72ch;
      color: var(--ink-soft);
      font-size: 1.05rem;
    }}
    .hero-badges {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .badge {{
      border: 1px solid rgba(15,118,110,0.2);
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 0.95rem;
      background: rgba(15,118,110,0.08);
      color: var(--accent-strong);
    }}
    .grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: 1.05fr 0.95fr;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 20px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 1.35rem;
      letter-spacing: -0.02em;
    }}
    .panel p, .panel li, .meta, label {{
      color: var(--ink-soft);
    }}
    .panel-error {{
      border-color: rgba(159,18,57,0.3);
      background: rgba(159,18,57,0.06);
    }}
    form {{
      display: grid;
      gap: 14px;
    }}
    .row {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    label {{
      display: grid;
      gap: 7px;
      font-size: 0.92rem;
    }}
    input, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: rgba(255,255,255,0.9);
    }}
    textarea {{
      min-height: 132px;
      resize: vertical;
    }}
    .actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    button {{
      appearance: none;
      border: none;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      padding: 12px 18px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      transition: transform 140ms ease, background 140ms ease;
    }}
    button:hover {{
      background: var(--accent-strong);
      transform: translateY(-1px);
    }}
    .muted {{
      color: var(--ink-soft);
      font-size: 0.92rem;
    }}
    .stats {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 14px;
    }}
    .stat {{
      border: 1px solid rgba(18,38,58,0.08);
      background: #faf6ef;
      border-radius: 16px;
      padding: 14px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.35rem;
      color: var(--ink);
      margin-bottom: 4px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #fbf8f2;
      border: 1px solid rgba(18,38,58,0.08);
      border-radius: 16px;
      padding: 14px;
      margin: 0;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      font-size: 0.9rem;
      line-height: 1.45;
      color: #213547;
    }}
    .chunk {{
      display: grid;
      gap: 8px;
      padding: 14px;
      border-radius: 16px;
      background: #fbf8f2;
      border: 1px solid rgba(18,38,58,0.08);
    }}
    .chunk + .chunk {{
      margin-top: 12px;
    }}
    .chunk-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      font-size: 0.9rem;
      color: var(--ink-soft);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(18,38,58,0.06);
      color: var(--ink);
    }}
    @media (max-width: 920px) {{
      .grid, .row, .stats {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        padding: 20px 14px 40px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>ESG RAG Studio</h1>
      <p>Build a searchable ESG knowledge base from PDF reports, store vectors in FAISS, inspect retrieval, and ask grounded questions through Albert from one local browser view.</p>
      <div class="hero-badges">
        <span class="badge">API key: {_escape(current_key_hint)}</span>
        <span class="badge">Index directory: {_escape(index_dir)}</span>
        <span class="badge">Sample PDF ready</span>
      </div>
    </section>

    {error_html}

    <section class="grid">
      <div class="stack">
        <section class="panel">
          <h2>Build Index</h2>
          <p>Paste one or more absolute PDF paths, one per line. Leave the sample path in place to rebuild the demo report index.</p>
          <form method="post" action="/build">
            <label>
              Albert API Key
              <input type="password" name="api_key" placeholder="Leave blank to use current ALBERT_API_KEY">
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
              <button type="submit">Build FAISS Index</button>
              <span class="muted">If FAISS is available, the backend will switch from NumPy to FAISS automatically.</span>
            </div>
          </form>
        </section>

        <section class="panel">
          <h2>Ask Question</h2>
          <p>Query the current index, inspect the retrieved chunks, and generate a grounded answer from the selected context.</p>
          <form method="post" action="/ask">
            <label>
              Albert API Key
              <input type="password" name="api_key" placeholder="Leave blank to use current ALBERT_API_KEY">
            </label>
            <label>
              ESG question
              <textarea name="question" style="min-height: 100px">What climate-related targets are disclosed for 2030, and are they science-based?</textarea>
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
              <button type="submit">Run Retrieval + Answer</button>
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
  <p>No built index found yet in <strong>{_escape(index_dir)}</strong>. Build one from the left panel to start querying.</p>
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
  <p class="meta">Built at: {_escape(summary.get("built_at", "unknown"))}</p>
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
