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


def main() -> None:
    st.set_page_config(page_title="ESG AI Prompt Console", layout="wide")

    st.session_state.setdefault("last_output", "")
    st.session_state.setdefault("uploaded_document", None)
    st.session_state.setdefault("document_context_enabled", True)

    flash_status = st.session_state.pop("flash_status", None)
    if flash_status:
        st.success(flash_status)

    st.title("Prompt Console")
    st.caption("Albert models only. Upload a document to answer from it.")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Albert API key",
            value=os.environ.get("ALBERT_API_KEY", ""),
            type="password",
        )
        albert_base_url = st.text_input("Albert base URL", value=DEFAULT_ALBERT_BASE_URL)

        models, warnings = load_model_catalog(api_key, base_url=albert_base_url)
        for warning in warnings:
            st.warning(warning)

        if models:
            selected_model = st.selectbox(
                "AI model",
                options=models,
                format_func=lambda model: model.label,
            )
        else:
            selected_model = None
            st.error("No Albert text-generation models were found.")

        st.subheader("Tuning")
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
            height=120,
        )

        st.divider()
        st.subheader("Document")
        company_hint = st.text_input(
            "Company name (optional)",
            placeholder="e.g. BNP Paribas",
            help="Used to match a report in sample_data when you do not upload a file.",
        )
        use_local_reports = st.checkbox(
            "Auto-select a report from sample_data",
            value=True,
            help="When no document is uploaded, the app looks for a PDF whose filename matches the company name in your prompt.",
        )
        uploaded_file = st.file_uploader(
            "Upload a document",
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
            st.caption(f"Loaded: {document['name']}")
            st.caption(f"{document['char_count']} characters extracted")

        if document is None:
            st.session_state["document_context_enabled"] = False

        st.checkbox(
            "Use uploaded document in prompt",
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
        st.info(f"Attached document: {document['name']} ({document['char_count']} characters)")

    with st.form("prompt_form", clear_on_submit=False):
        prompt = st.text_area(
            "Prompt",
            height=220,
            placeholder="Ask a question, draft a note, or request an analysis.",
        )
        submitted = st.form_submit_button("Run")

    if submitted:
        if not selected_model:
            st.error("Select an Albert model first.")
        elif not prompt.strip():
            st.warning("Type a prompt before running the model.")
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
                    selected_document_note = f"Matched local report: {selected_path.name} (score {match_score:.1f})"
                elif local_reports:
                    selected_document_note = "No local report matched the company name in the prompt."

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
                    st.success(f"Completed using {selected_model.label}.")
                    if selected_document_note:
                        st.caption(selected_document_note)

    st.subheader("Output")
    output_text = st.session_state.get("last_output", "")
    with st.container(border=True):
        if output_text:
            st.markdown(output_text)
        else:
            st.caption("Run a prompt to see the rendered markdown output here.")

    if st.session_state.get("last_prompt_with_context"):
        with st.expander("Prompt sent to the model", expanded=False):
            st.markdown(st.session_state.get("last_prompt_with_context", ""))


if __name__ == "__main__":
    main()