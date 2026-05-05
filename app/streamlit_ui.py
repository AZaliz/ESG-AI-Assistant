"""Simple Streamlit prompt console for Albert models."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from rapidfuzz import fuzz

from app.llm import DEFAULT_ALBERT_BASE_URL, ChatModel, generate_completion, load_model_catalog
from app.parsers.html_parser import parse_html
from app.parsers.pdf_parser import parse_pdf


SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".md"}
DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful assistant. Answer directly and clearly."
LOCAL_REPORT_LIBRARY = Path(__file__).resolve().parent.parent / "sample_data"


def _humanize_path_stem(path: Path) -> str:
    return re.sub(r"\s+", " ", path.stem.replace("_", " ").replace("-", " ")).strip()


def _discover_local_reports() -> list[Path]:
    if not LOCAL_REPORT_LIBRARY.exists():
        return []
    return sorted(path for path in LOCAL_REPORT_LIBRARY.glob("*.pdf") if path.is_file())


def _score_report_match(prompt: str, company_hint: str, report_path: Path) -> float:
    prompt_text = prompt.lower().strip()
    company_text = company_hint.lower().strip()
    report_text = _humanize_path_stem(report_path).lower()
    prompt_terms = set(re.findall(r"[a-z0-9]+", prompt_text))
    company_terms = set(re.findall(r"[a-z0-9]+", company_text))
    report_terms = set(re.findall(r"[a-z0-9]+", report_text))

    shared_prompt_terms = prompt_terms & report_terms
    shared_company_terms = company_terms & report_terms

    score = float(len(shared_prompt_terms) * 100 + len(shared_company_terms) * 150)
    score += max(
        float(fuzz.partial_ratio(prompt_text, report_text) / 5.0),
        float(fuzz.token_set_ratio(prompt_text, report_text) / 5.0),
    )
    if company_text:
        score += float(fuzz.partial_ratio(company_text, report_text) / 10.0)
        score += float(fuzz.token_set_ratio(company_text, report_text) / 10.0)
        if company_terms and company_terms <= report_terms:
            score += 100.0
    return score


def _select_local_report(prompt: str, company_hint: str, report_paths: list[Path]) -> tuple[Path | None, float]:
    best_path: Path | None = None
    best_score = 0.0
    for report_path in report_paths:
        score = _score_report_match(prompt, company_hint, report_path)
        if score > best_score:
            best_path = report_path
            best_score = score

    if best_path and best_score >= 55.0:
        return best_path, best_score
    return None, 0.0


def _read_document_path(document_path: Path, *, display_name: str | None = None) -> dict[str, Any]:
    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise RuntimeError("Unsupported file type. Upload a PDF, HTML, Markdown, or text file.")

    if suffix == ".pdf":
        parsed = parse_pdf(document_path)
        text = parsed.text.strip()
        summary = parsed.notes or "Document uploaded successfully."
        page_count = parsed.page_count or 0
    elif suffix in {".html", ".htm"}:
        parsed = parse_html(document_path)
        text = parsed.text.strip()
        summary = parsed.notes or "Document uploaded successfully."
        page_count = parsed.page_count or 0
    else:
        text = document_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise RuntimeError("The uploaded text file was empty.")
        summary = "Document uploaded successfully."
        page_count = 0

    if not text:
        raise RuntimeError("No extractable text was found in the uploaded document.")

    return {
        "name": display_name or document_path.name,
        "suffix": suffix,
        "text": text,
        "page_count": page_count,
        "summary": summary,
        "char_count": len(text),
        "source_path": str(document_path),
    }


def _read_uploaded_document(uploaded_file: Any) -> dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded_file.getvalue())
            temp_path = Path(handle.name)
        return _read_document_path(temp_path, display_name=uploaded_file.name)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _chunk_document_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise RuntimeError("Chunk size must be greater than zero.")
    if chunk_overlap < 0:
        raise RuntimeError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise RuntimeError("Chunk overlap must be smaller than chunk size.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _build_document_prompt(
    prompt: str,
    document: dict[str, Any],
    *,
    chunk_size: int,
    chunk_overlap: int,
    company_hint: str = "",
) -> str:
    document_text = str(document.get("text", ""))
    chunks = _chunk_document_text(document_text, chunk_size, chunk_overlap)
    if not chunks:
        raise RuntimeError("No extractable text was found in the uploaded document.")

    context_budget = 12000
    context_parts: list[str] = []
    used_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        section = f"Chunk {index}/{len(chunks)}:\n{chunk}"
        if used_chars + len(section) > context_budget:
            remaining = len(chunks) - index + 1
            if remaining > 0:
                context_parts.append(f"[{remaining} additional chunk(s) omitted to keep the prompt short.]")
            break
        context_parts.append(section)
        used_chars += len(section)

    context_block = "\n\n".join(context_parts)

    return (
        "Use the uploaded document as the main source of truth. If the answer is not in the document, say so clearly.\n\n"
        f"Target company: {company_hint.strip() or 'auto-detected from prompt'}\n"
        f"Document: {document.get('name', 'uploaded file')}\n"
        f"Source path: {document.get('source_path', '')}\n"
        f"Summary: {document.get('summary', '')}\n"
        f"Pages: {document.get('page_count', 0)}\n\n"
        f"Chunk size: {chunk_size} words\n"
        f"Chunk overlap: {chunk_overlap} words\n\n"
        f"Document chunks:\n{context_block}\n\n"
        f"Question:\n{prompt.strip()}"
    )


def _inject_terminal_css() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

    * { font-family: "JetBrains Mono", "IBM Plex Mono", "SF Mono", "Menlo", monospace !important; }

    .stApp { background: #07080e; }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1100px;
    }

    section[data-testid="stSidebar"] {
        background: #0a0c15;
        border-right: 1px solid #1a1c2e;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    h1, h2, h3, h4 { color: #e1e4f0 !important; letter-spacing: -0.01em; }
    h1 { font-size: 1.5rem !important; font-weight: 600 !important; }
    h2 { font-size: 0.95rem !important; font-weight: 500 !important; }
    h3 { font-size: 0.82rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.06em; color: #8b8fa8 !important; }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: #0a0b14 !important;
        border: 1px solid #1a1c2e !important;
        border-radius: 6px !important;
        color: #e1e4f0 !important;
        font-size: 0.78rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 2px rgba(16,185,129,0.18) !important;
    }

    .stButton > button {
        background: rgba(16,185,129,0.1) !important;
        border: 1px solid #10b981 !important;
        border-radius: 6px !important;
        color: #10b981 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em;
        transition: background 140ms ease, box-shadow 140ms ease;
    }
    .stButton > button:hover {
        background: rgba(16,185,129,0.18) !important;
        box-shadow: 0 0 12px rgba(16,185,129,0.18) !important;
        border-color: #10b981 !important;
        color: #10b981 !important;
    }

    .stFormSubmitButton > button { border-width: 1px !important; }

    .stDownloadButton > button {
        background: rgba(34,211,238,0.1) !important;
        border: 1px solid #22d3ee !important;
        color: #22d3ee !important;
        border-radius: 6px !important;
    }

    .stSlider > div > div > div > div { background: #10b981 !important; }

    .stCheckbox > label > div[data-baseweb="checkbox"] > div { border-color: #1a1c2e !important; }
    .stCheckbox > label > div[data-baseweb="checkbox"][data-checked="true"] > div { background: #10b981 !important; border-color: #10b981 !important; }

    div[data-testid="stExpander"] { border: 1px solid #1a1c2e !important; border-radius: 10px !important; }
    div[data-testid="stExpander"] > div { background: #0d0f1a !important; }
    
    .stAlert { border-radius: 6px !important; }
    div[data-baseweb="select"] > div { background: #0a0b14 !important; border-color: #1a1c2e !important; }
    
    div[data-testid="stCaptionContainer"] { color: #5c607a !important; font-size: 0.7rem; }

    .stFileUploader > section > div { border: 1px dashed #1a1c2e !important; border-radius: 6px !important; background: #0a0b14 !important; }
    .stFileUploader > section > div:hover { border-color: #10b981 !important; }

    hr { border-color: #1a1c2e !important; }

    label, .stMarkdown p, .stMarkdown li { color: #8b8fa8 !important; font-size: 0.75rem; }
    label > div:first-child { color: #5c607a !important; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.7rem !important; }

    .stSpinner > div { border-color: #10b981 !important; }

    pre, code {
        background: #0a0b14 !important;
        border: 1px solid #1a1c2e !important;
        border-radius: 6px;
        font-size: 0.78rem !important;
        color: #8b8fa8 !important;
    }

    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #07080e; }
    ::-webkit-scrollbar-thumb { background: #1e2035; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2a2d45; }
    </style>""", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="ESG AI Terminal", layout="wide")
    _inject_terminal_css()

    st.session_state.setdefault("last_output", "")
    st.session_state.setdefault("uploaded_document", None)
    st.session_state.setdefault("document_context_enabled", True)

    flash_status = st.session_state.pop("flash_status", None)
    if flash_status:
        st.success(flash_status)

    st.markdown('<h1 style="margin:0;padding:0;">ESG_AI <span style="color:#10b981;">v1.0</span></h1>', unsafe_allow_html=True)
    st.caption("> Ask questions against ESG reports using Albert models. Upload a document or use local sample_data.")

    with st.sidebar:
        st.markdown("### > settings")
        api_key = st.text_input(
            "API Key",
            value=os.environ.get("ALBERT_API_KEY", ""),
            type="password",
            placeholder="sk-...",
        )
        albert_base_url = st.text_input("Base URL", value=DEFAULT_ALBERT_BASE_URL)

        models, warnings = load_model_catalog(api_key, base_url=albert_base_url)
        for warning in warnings:
            st.warning(warning)

        if models:
            selected_model = st.selectbox(
                "Model",
                options=models,
                format_func=lambda model: model.label,
            )
        else:
            selected_model = None
            st.error("No Albert text-generation models found.")

        st.markdown("### > tuning")
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
        top_k = st.slider("Top K", min_value=1, max_value=100, value=40, step=1)
        chunk_size = st.slider("Chunk size (words)", min_value=100, max_value=2000, value=400, step=50)
        chunk_overlap = st.slider(
            "Chunk overlap (words)",
            min_value=0,
            max_value=max(0, chunk_size - 1),
            value=min(80, max(0, chunk_size // 4)),
            step=10,
        )
        system_prompt = st.text_area(
            "System prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=100,
        )

        st.markdown("---")
        st.markdown("### > document")
        company_hint = st.text_input(
            "Company (optional)",
            placeholder="e.g. BNP Paribas",
            help="Used to match a report in sample_data when you do not upload a file.",
        )
        use_local_reports = st.checkbox(
            "Auto-select from sample_data",
            value=True,
            help="When no document is uploaded, the app matches a local PDF by company name.",
        )
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "txt", "md", "html", "htm"],
        )

        document = st.session_state.get("uploaded_document")

        if uploaded_file is not None and st.button("Load document"):
            try:
                document = _read_uploaded_document(uploaded_file)
            except Exception as exc:  # noqa: BLE001
                st.session_state["flash_status"] = f"Could not load document: {exc}"
                st.rerun()
            else:
                st.session_state["uploaded_document"] = document
                st.session_state["document_context_enabled"] = True
                st.session_state["flash_status"] = (
                    f"Loaded {document['name']} ({document['char_count']} characters)."
                )
                st.rerun()

        document = st.session_state.get("uploaded_document")
        if document:
            st.caption(f"Loaded: {document['name']} ({document['char_count']} chars)")

        if document is None:
            st.session_state["document_context_enabled"] = False

        st.checkbox(
            "Use document in prompt",
            key="document_context_enabled",
            disabled=document is None,
        )

        if document and st.button("Clear document"):
            st.session_state.pop("uploaded_document", None)
            st.session_state["document_context_enabled"] = False
            st.session_state["flash_status"] = "Document cleared."
            st.rerun()

    if st.session_state.get("uploaded_document"):
        document = st.session_state["uploaded_document"]
        st.info(f"Document attached: {document['name']} ({document['char_count']} characters)")

    with st.form("prompt_form", clear_on_submit=False):
        prompt = st.text_area(
            "Prompt",
            height=180,
            placeholder="> Ask a question, draft analysis, or request a summary...",
        )
        submitted = st.form_submit_button("Execute")

    if submitted:
        if not selected_model:
            st.error("Select an Albert model first.")
        elif not prompt.strip():
            st.warning("Type a prompt before running.")
        else:
            final_prompt = prompt.strip()
            selected_document = None
            selected_document_note = ""
            if st.session_state.get("document_context_enabled") and st.session_state.get("uploaded_document"):
                selected_document = st.session_state["uploaded_document"]
                selected_document_note = f"Uploaded document: {selected_document.get('name', 'uploaded file')}"
            elif use_local_reports:
                local_reports = _discover_local_reports()
                selected_path, match_score = _select_local_report(prompt, company_hint, local_reports)
                if selected_path:
                    selected_document = _read_document_path(selected_path)
                    selected_document["match_score"] = match_score
                    selected_document_note = f"Matched report: {selected_path.name} (score {match_score:.1f})"
                elif local_reports:
                    selected_document_note = "No local report matched the company name."

            if selected_document:
                final_prompt = _build_document_prompt(
                    prompt,
                    selected_document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    company_hint=company_hint,
                )
                if selected_document_note:
                    final_prompt = f"{selected_document_note}\n\n{final_prompt}"

            with st.spinner(f"Running {selected_model.label}..."):
                try:
                    response_text = generate_completion(
                        selected_model,
                        prompt=final_prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        top_k=top_k,
                        api_key=api_key,
                        base_url=albert_base_url,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["last_output"] = f"Error: {exc}"
                    st.error(exc)
                else:
                    st.session_state["last_output"] = response_text
                    st.session_state["last_prompt_with_context"] = final_prompt
                    st.session_state["last_model"] = selected_model.label
                    st.success(f"Completed ({selected_model.label}).")
                    if selected_document_note:
                        st.caption(selected_document_note)

    st.markdown("### > output")
    output_text = st.session_state.get("last_output", "")
    if output_text:
        st.markdown(output_text)
    else:
        st.caption("Execute a prompt to see output here.")

    if st.session_state.get("last_prompt_with_context"):
        with st.expander("Full prompt sent to model"):
            st.markdown(st.session_state.get("last_prompt_with_context", ""))


if __name__ == "__main__":
    main()