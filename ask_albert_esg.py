#!/usr/bin/env python3
"""Ask one-off questions against an Albert collection using native RAG search."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_COLLECTION_NAME = "esg-disclosures"


@dataclass
class AlbertClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 120

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models").json().get("data", [])

    def get_text_generation_model(self) -> str:
        for model in self.list_models():
            if model.get("type") == "text-generation":
                return str(model["id"])
        raise RuntimeError("No Albert text-generation model was returned by /models.")

    def find_collection_by_name(self, name: str) -> dict[str, Any] | None:
        response = self._request("GET", "/collections", params={"name": name}).json()
        for collection in response.get("data", []):
            if collection.get("name") == name:
                return collection
        return None

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        collection_id: int | str,
        limit: int,
        method: str,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "tools": [
                {
                    "type": "search",
                    "collection_ids": [int(collection_id)],
                    "method": method,
                    "limit": limit,
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        }
        return self._request("POST", "/chat/completions", json=payload).json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask one-off questions against an Albert ESG collection."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to ask about the uploaded ESG documents.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Albert collection name to query.",
    )
    parser.add_argument(
        "--collection-id",
        type=int,
        help="Albert collection id to query. Overrides --collection-name.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=5,
        help="How many chunks Albert can retrieve while answering.",
    )
    parser.add_argument(
        "--search-method",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Albert SearchTool retrieval method.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw Albert JSON response instead of only the answer text.",
    )
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.environ.get("ALBERT_API_KEY")
    if not api_key:
        raise RuntimeError("Set ALBERT_API_KEY in your environment before running this script.")
    return api_key


def extract_message_content(chat_response: dict[str, Any]) -> str:
    choices = chat_response.get("choices") or []
    if not choices:
        raise RuntimeError("Albert returned no choices in chat completion response.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(text_parts).strip()
    raise RuntimeError("Unable to extract assistant content from chat completion response.")


def resolve_collection_id(client: AlbertClient, collection_id: int | None, collection_name: str) -> int:
    if collection_id is not None:
        return collection_id

    collection = client.find_collection_by_name(collection_name)
    if not collection:
        raise RuntimeError(
            f"Collection '{collection_name}' was not found. Run esg_rag_runner.py first or pass --collection-id."
        )
    return int(collection["id"])


def main() -> int:
    args = parse_args()
    client = AlbertClient(
        api_key=require_api_key(),
        base_url=os.environ.get("ALBERT_BASE_URL", DEFAULT_BASE_URL),
    )
    model = client.get_text_generation_model()
    collection_id = resolve_collection_id(client, args.collection_id, args.collection_name)
    question = " ".join(args.question).strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an ESG analyst. Answer using only retrieved evidence from the indexed documents. "
                "If the evidence is insufficient, say so clearly."
            ),
        },
        {"role": "user", "content": question},
    ]

    response = client.chat_completion(
        model=model,
        messages=messages,
        collection_id=collection_id,
        limit=args.search_limit,
        method=args.search_method,
    )

    if args.json:
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print(extract_message_content(response))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
