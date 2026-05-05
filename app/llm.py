"""Albert chat helpers used by the Streamlit prompt console."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"


@dataclass(frozen=True)
class ChatModel:
    model_id: str
    label: str


def _albert_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


def require_albert_api_key(api_key: str | None = None) -> str:
    resolved = (api_key or os.environ.get("ALBERT_API_KEY") or "").strip()
    if not resolved:
        raise RuntimeError("Set ALBERT_API_KEY before using Albert models.")
    return resolved


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in text_parts if part).strip()

    message = payload.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(part for part in text_parts if part).strip()

    raise RuntimeError("Unexpected chat completion payload.")


def list_albert_models(api_key: str | None, base_url: str = DEFAULT_ALBERT_BASE_URL) -> list[ChatModel]:
    resolved_key = (api_key or os.environ.get("ALBERT_API_KEY") or "").strip()
    if not resolved_key:
        return []

    session = _albert_session(resolved_key)
    response = session.get(f"{base_url.rstrip('/')}/models", timeout=30)
    response.raise_for_status()

    models: list[ChatModel] = []
    for model in response.json().get("data", []):
        if str(model.get("type", "")).lower() != "text-generation":
            continue
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            continue
        models.append(ChatModel(model_id=model_id, label=f"Albert | {model_id}"))

    models.sort(key=lambda item: item.model_id.lower())
    return models


def load_model_catalog(api_key: str | None, *, base_url: str = DEFAULT_ALBERT_BASE_URL) -> tuple[list[ChatModel], list[str]]:
    warnings: list[str] = []
    resolved_key = (api_key or os.environ.get("ALBERT_API_KEY") or "").strip()
    if not resolved_key:
        warnings.append("Enter an Albert API key to load available models.")
        return [], warnings

    try:
        models = list_albert_models(resolved_key, base_url=base_url)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Unable to load Albert models: {exc}")
        return [], warnings

    if not models:
        warnings.append("Albert returned no text-generation models.")
    return models, warnings


def build_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def generate_completion(
    model: ChatModel,
    *,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_k: int,
    api_key: str | None,
    base_url: str = DEFAULT_ALBERT_BASE_URL,
) -> str:
    resolved_key = require_albert_api_key(api_key)
    session = _albert_session(resolved_key)
    payload: dict[str, Any] = {
        "model": model.model_id,
        "messages": build_messages(prompt, system_prompt),
        "stream": False,
        "temperature": temperature,
        "top_k": top_k,
    }
    response = session.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=120)
    response.raise_for_status()
    return _extract_message_content(response.json())