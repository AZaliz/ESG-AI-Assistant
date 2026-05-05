"""Model registry and chat providers for the Streamlit prompt playground."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_FREE_MODELS: tuple[tuple[str, str], ...] = (
    ("qwen2.5:7b", "Qwen2.5 7B"),
    ("llama3.1:8b", "Llama 3.1 8B"),
)


@dataclass(frozen=True)
class ChatModel:
    provider: str
    model_id: str
    label: str
    available: bool = True

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model_id}"


def _dedupe_models(models: list[ChatModel]) -> list[ChatModel]:
    seen: set[str] = set()
    unique: list[ChatModel] = []
    for model in models:
        if model.key in seen:
            continue
        seen.add(model.key)
        unique.append(model)
    return unique


def _albert_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


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
    if not api_key:
        return []

    session = _albert_session(api_key.strip())
    response = session.get(f"{base_url.rstrip('/')}/models", timeout=30)
    response.raise_for_status()

    models: list[ChatModel] = []
    for model in response.json().get("data", []):
        if str(model.get("type", "")).lower() != "text-generation":
            continue
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            continue
        models.append(ChatModel(provider="albert", model_id=model_id, label=f"Albert | {model_id}"))

    return _dedupe_models(sorted(models, key=lambda item: item.model_id.lower()))


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> list[ChatModel]:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=30)
    response.raise_for_status()

    models: list[ChatModel] = []
    for model in response.json().get("models", []):
        model_id = str(model.get("name", "")).strip()
        if not model_id:
            continue
        models.append(ChatModel(provider="ollama", model_id=model_id, label=f"Ollama | {model_id}"))

    return _dedupe_models(sorted(models, key=lambda item: item.model_id.lower()))


def suggest_free_fallbacks(models: list[ChatModel]) -> list[ChatModel]:
    if len(models) > 1:
        return []

    existing = {model.key for model in models}
    suggestions: list[ChatModel] = []
    for model_id, label in DEFAULT_FREE_MODELS:
        fallback = ChatModel(
            provider="ollama",
            model_id=model_id,
            label=f"Free local | {label}",
            available=False,
        )
        if fallback.key not in existing:
            suggestions.append(fallback)
    return suggestions


def load_model_catalog(
    api_key: str | None,
    *,
    albert_base_url: str = DEFAULT_ALBERT_BASE_URL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> tuple[list[ChatModel], list[ChatModel], list[str]]:
    warnings: list[str] = []
    models: list[ChatModel] = []

    try:
        models.extend(list_albert_models(api_key, base_url=albert_base_url))
    except Exception as exc:  # noqa: BLE001
        if api_key:
            warnings.append(f"Albert model lookup failed: {exc}")
        else:
            warnings.append("ALBERT_API_KEY is missing, so Albert models were skipped.")

    try:
        models.extend(list_ollama_models(base_url=ollama_base_url))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Ollama is not available yet: {exc}")

    models = _dedupe_models(models)
    suggestions = suggest_free_fallbacks(models)
    return models, suggestions, warnings


def download_ollama_model(model_id: str) -> str:
    command = shutil.which("ollama")
    if not command:
        raise RuntimeError("Ollama is not installed. Install it to download a free local fallback model.")

    completed = subprocess.run([command, "pull", model_id], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"Failed to download {model_id}."
        raise RuntimeError(message)

    return completed.stdout.strip() or completed.stderr.strip() or f"Downloaded {model_id}."


def build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_prompt = system_prompt.strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
    albert_base_url: str = DEFAULT_ALBERT_BASE_URL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> str:
    messages = build_messages(prompt, system_prompt)

    if model.provider == "albert":
        if not api_key:
            raise RuntimeError("Set ALBERT_API_KEY or enter it in the sidebar before using Albert models.")

        session = _albert_session(api_key.strip())
        payload: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "top_k": top_k,
        }
        response = session.post(f"{albert_base_url.rstrip('/')}/chat/completions", json=payload, timeout=120)
        response.raise_for_status()
        return _extract_message_content(response.json())

    if model.provider == "ollama":
        payload: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_k": top_k,
            },
        }
        response = requests.post(f"{ollama_base_url.rstrip('/')}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        return _extract_message_content(response.json())

    raise RuntimeError(f"Unsupported provider '{model.provider}'.")