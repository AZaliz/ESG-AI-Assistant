"""Streamlit prompt playground with a CLI-like terminal appearance."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from app.llm import (
    ChatModel,
    DEFAULT_ALBERT_BASE_URL,
    DEFAULT_OLLAMA_BASE_URL,
    download_ollama_model,
    generate_completion,
    load_model_catalog,
)
from app.parsers.html_parser import parse_html
from app.parsers.pdf_parser import parse_pdf


SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".md"}


APP_TITLE = "opencode / ESG AI"
DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful assistant. Answer directly and clearly."


def _inject_css() -> None:
    st.markdown(
        """
<style>
  :root {
    --bg: #06090f;
    --panel: #0d141d;
    --panel-2: #111a24;
    --line: #223041;
    --text: #d5e0ea;
    --muted: #7f90a3;
    --accent: #7df9aa;
    --accent-2: #4ee0c4;
    --danger: #ff6b6b;
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
  }

  .stApp {
    background:
      radial-gradient(circle at top left, rgba(125, 249, 170, 0.12), transparent 26%),
      radial-gradient(circle at top right, rgba(78, 224, 196, 0.10), transparent 22%),
      linear-gradient(180deg, #06090f 0%, #0a1018 100%);
    color: var(--text);
    font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
  }

  #MainMenu, footer, header {
    visibility: hidden;
  }

  .block-container {
    padding-top: 1.1rem;
    padding-bottom: 1.6rem;
  }

  [data-testid="stSidebar"] {
    background: rgba(7, 11, 17, 0.96);
    border-right: 1px solid var(--line);
  }

  h1, h2, h3, h4, label, input, textarea, button {
    font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace !important;
  }

  .terminal-shell {
    background: rgba(10, 16, 24, 0.92);
    border: 1px solid var(--line);
    border-radius: 22px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }

  .terminal-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    padding: 1rem 1.2rem;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(17, 26, 36, 0.96), rgba(13, 20, 29, 0.96));
  }

  .terminal-title {
    margin: 0;
    font-size: clamp(1.8rem, 4vw, 3rem);
    line-height: 1;
    letter-spacing: -0.06em;
    color: var(--text);
  }

  .terminal-subtitle {
    margin: 0.45rem 0 0;
    color: var(--muted);
    max-width: 72ch;
    font-size: 0.95rem;
    line-height: 1.5;
  }

  .terminal-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border: 1px solid rgba(125, 249, 170, 0.22);
    border-radius: 999px;
    padding: 0.38rem 0.75rem;
    color: var(--accent);
    background: rgba(125, 249, 170, 0.08);
    font-size: 0.74rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .terminal-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: flex-end;
  }

  .status-chip {
    border: 1px solid rgba(78, 224, 196, 0.22);
    border-radius: 999px;
    padding: 0.42rem 0.75rem;
    color: var(--text);
    background: rgba(78, 224, 196, 0.08);
    font-size: 0.78rem;
  }

  .terminal-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
    align-items: start;
    padding: 1rem;
  }

  .terminal-panel {
    background: rgba(13, 20, 29, 0.96);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem;
  }

  .terminal-panel h2 {
    margin: 0 0 0.65rem;
    color: var(--accent);
    font-size: 0.88rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .terminal-panel p {
    margin-top: 0;
    color: var(--muted);
    line-height: 1.5;
  }

  .small-muted {
    color: var(--muted);
    font-size: 0.85rem;
  }

  .stTextArea textarea,
  .stTextInput input,
  .stSelectbox div[data-baseweb="select"],
  .stNumberInput input {
    background: #081019 !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
  }

  .stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #02120a;
    border: none;
    border-radius: 999px;
    font-weight: 700;
    padding: 0.75rem 1rem;
  }

  .stButton button:hover {
    filter: brightness(1.02);
  }

  @media (max-width: 960px) {
    .terminal-grid {
      grid-template-columns: 1fr;
    }

    .terminal-header {
      flex-direction: column;
    }

    .terminal-meta {
      justify-content: flex-start;
    }
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(model_count: int, fallback_count: int) -> None:
    st.markdown(
        f"""
<section class="terminal-shell">
  <div class="terminal-header">
    <div>
      <div class="terminal-badge">{APP_TITLE}</div>
      <h1 class="terminal-title">Prompt Console</h1>
      <p class="terminal-subtitle">
        A Streamlit playground with a CLI feel: pick a model, tune temperature and top-k,
        type a prompt, and read the response in a terminal-style output panel.
      </p>
    </div>
    <div class="terminal-meta">
      <span class="status-chip">{model_count} available model(s)</span>
      <span class="status-chip">{fallback_count} free fallback option(s)</span>
      <span class="status-chip">Albert + Ollama</span>
    </div>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


def _select_model(models: list[ChatModel]) -> ChatModel | None:
    if not models:
        return None
    return st.selectbox(
        "AI model",
        options=models,
        format_func=lambda model: model.label,
        help="Available chat models from Albert and local Ollama installs.",
    )


def _render_fallbacks(fallback_models: list[ChatModel]) -> None:
    if not fallback_models:
        return

    st.markdown(
        """
<div class="terminal-panel">
  <h2>Free fallback</h2>
  <p>Only one model is available. Pull a local Ollama model to create a second choice.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    for fallback in fallback_models:
        if st.button(f"Download {fallback.model_id}", key=f"pull-{fallback.model_id}"):
            with st.spinner(f"Downloading {fallback.model_id} ..."):
                result = download_ollama_model(fallback.model_id)
            st.session_state["flash_status"] = result
            st.rerun()


def _read_uploaded_document(uploaded_file: Any) -> dict[str, str | int]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise RuntimeError("Unsupported file type. Upload a PDF, HTML, Markdown, or text file.")

    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getvalue())
        temp_path = Path(handle.name)

    try:
        if suffix == ".pdf":
            parse_result = parse_pdf(temp_path)
        elif suffix in {".html", ".htm"}:
            parse_result = parse_html(temp_path)
        else:
            text = temp_path.read_text(encoding="utf-8", errors="ignore").strip()
            parse_result = None
            if not text:
                raise RuntimeError("The uploaded text file was empty.")
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    if suffix in {".html", ".htm"} or suffix == ".pdf":
        text = parse_result.text.strip() if parse_result and parse_result.text else ""

    if not text:
        raise RuntimeError("No extractable text was found in the uploaded document.")

    page_count = int(parse_result.page_count) if parse_result and parse_result.page_count else 0
    summary = parse_result.notes if parse_result and parse_result.notes else "Document uploaded successfully."
    return {
        "name": uploaded_file.name,
        "suffix": suffix,
        "text": text,
        "page_count": page_count,
        "summary": summary,
        "char_count": len(text),
    }


def _build_document_prompt(prompt: str, document: dict[str, str | int]) -> str:
    document_text = str(document.get("text", ""))
    char_limit = 20000
    if len(document_text) > char_limit:
        document_text = document_text[:char_limit] + "\n\n[Document truncated for context length]"

    return (
        "Use the uploaded document as primary context. If the document does not contain the answer, say so clearly. "
        "Cite the document when helpful.\n\n"
        f"Document: {document.get('name', 'uploaded file')}\n"
        f"Type: {document.get('suffix', '')}\n"
        f"Pages: {document.get('page_count', 0)}\n"
        f"Summary: {document.get('summary', '')}\n\n"
        f"Document content:\n{document_text}\n\n"
        f"User question:\n{prompt.strip()}"
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=">", layout="wide", initial_sidebar_state="expanded")
    _inject_css()

    flash_status = st.session_state.pop("flash_status", None)
    if flash_status:
        st.success(flash_status)

    st.session_state.setdefault("uploaded_document", None)
    st.session_state.setdefault("document_context_enabled", True)

    api_key = st.sidebar.text_input(
        "Albert API key",
        value=os.environ.get("ALBERT_API_KEY", ""),
        type="password",
        help="Used for Albert-hosted models. Leave blank if you only want local Ollama models.",
    )
    albert_base_url = st.sidebar.text_input("Albert base URL", value=DEFAULT_ALBERT_BASE_URL)
    ollama_base_url = st.sidebar.text_input("Ollama base URL", value=DEFAULT_OLLAMA_BASE_URL)

    models, fallback_models, warnings = load_model_catalog(
        api_key,
        albert_base_url=albert_base_url,
        ollama_base_url=ollama_base_url,
    )

    st.sidebar.markdown("### Tuning")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=100, value=40, step=1)

    st.sidebar.markdown("### System Prompt")
    system_prompt = st.sidebar.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=110,
        help="Optional instruction that is prepended before the user prompt.",
    )

    st.sidebar.markdown("### Document")
    uploaded_file = st.sidebar.file_uploader(
      "Upload a document",
      type=["pdf", "txt", "md", "html", "htm"],
      help="Attach a document so the model can answer questions from it.",
    )
    use_uploaded_document = st.sidebar.checkbox(
      "Use uploaded document in prompt",
      value=bool(st.session_state.get("uploaded_document")),
      help="Include the uploaded file as context for the next query.",
    )
    st.session_state["document_context_enabled"] = use_uploaded_document

    if uploaded_file is not None and st.sidebar.button("Load document"):
      try:
        document_payload = _read_uploaded_document(uploaded_file)
      except Exception as exc:  # noqa: BLE001
        st.session_state["flash_status"] = f"Document load failed: {exc}"
        st.rerun()
      else:
        st.session_state["uploaded_document"] = document_payload
        st.session_state["flash_status"] = (
          f"Loaded {document_payload['name']} with {document_payload['char_count']} characters."
        )
        st.rerun()

    if st.session_state.get("uploaded_document"):
      document = st.session_state["uploaded_document"]
      st.sidebar.success(f"Loaded: {document['name']}")
      st.sidebar.caption(f"{document['char_count']} characters extracted")
      if st.sidebar.button("Clear document"):
        st.session_state.pop("uploaded_document", None)
        st.session_state.pop("last_prompt", None)
        st.session_state.pop("last_output", None)
        st.session_state["flash_status"] = "Document cleared."
        st.rerun()

    if warnings:
        for warning in warnings:
            st.sidebar.warning(warning)

    if models:
        selected_model = _select_model(models)
    else:
        selected_model = None
        st.sidebar.error("No models could be loaded yet.")

    if len(models) <= 1:
        _render_fallbacks(fallback_models)

    _render_header(model_count=len(models), fallback_count=len(fallback_models))

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            """
<div class="terminal-panel">
  <h2>Prompt</h2>
  <p>Type a prompt below, then run it with the selected model. When a document is loaded, the model uses it as context.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("prompt-form", clear_on_submit=False):
            prompt = st.text_area(
                "Prompt",
                placeholder="Ask a question, draft a note, or request an analysis...",
                height=220,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Run")

        if st.session_state.get("uploaded_document"):
            document = st.session_state["uploaded_document"]
            st.markdown(
                f"""
<div class="terminal-panel">
  <h2>Attached Document</h2>
  <p><strong>{document['name']}</strong></p>
  <p>{document['char_count']} characters extracted.</p>
</div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            """
<div class="terminal-panel">
  <h2>Output</h2>
  <p>Response text appears here after you run the prompt.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.text_area(
            "Output",
            value=st.session_state.get("last_output", ""),
            height=350,
            label_visibility="collapsed",
            disabled=True,
        )

    if submitted:
      if not selected_model:
        st.session_state["flash_status"] = "Load at least one model before running a prompt."
        st.rerun()

      if not prompt.strip():
        st.warning("Type a prompt before running the model.")
      else:
        final_prompt = prompt
        if st.session_state.get("document_context_enabled") and st.session_state.get("uploaded_document"):
          final_prompt = _build_document_prompt(prompt, st.session_state["uploaded_document"])

        started_at = time.perf_counter()
        with st.spinner(f"Running {selected_model.label} ..."):
          try:
            response_text = generate_completion(
              selected_model,
              prompt=final_prompt,
              system_prompt=system_prompt,
              temperature=temperature,
              top_k=top_k,
              api_key=api_key,
              albert_base_url=albert_base_url,
              ollama_base_url=ollama_base_url,
            )
          except Exception as exc:  # noqa: BLE001
            st.session_state["last_output"] = f"Error: {exc}"
            st.session_state["flash_status"] = f"Error: {exc}"
            st.rerun()
          else:
            elapsed = time.perf_counter() - started_at
            st.session_state["last_output"] = response_text
            st.session_state["last_model"] = selected_model.label
            st.session_state["last_prompt"] = prompt
            st.session_state["last_prompt_with_context"] = final_prompt
            st.session_state["last_tuning"] = {"temperature": temperature, "top_k": top_k}
            st.session_state["flash_status"] = f"Completed in {elapsed:.2f}s using {selected_model.label}."
            st.rerun()

    if st.session_state.get("last_output"):
        with st.expander("Last run details", expanded=False):
            st.write(f"Model: {st.session_state.get('last_model', 'unknown')}")
            tuning = st.session_state.get("last_tuning", {})
            st.write(f"Temperature: {tuning.get('temperature', temperature)}")
            st.write(f"Top K: {tuning.get('top_k', top_k)}")
            st.text_area(
                "Prompt history",
                value=st.session_state.get("last_prompt", ""),
                height=150,
                disabled=True,
            )
            st.text_area(
                "Prompt with document context",
                value=st.session_state.get("last_prompt_with_context", ""),
                height=180,
                disabled=True,
            )


if __name__ == "__main__":
    main()