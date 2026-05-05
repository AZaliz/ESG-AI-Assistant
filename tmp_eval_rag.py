import os
import sys
import csv
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.rag import AlbertClient, build_index, retrieve_chunks

DATASET_PATH = Path("/Users/ems/Desktop/ESG_AI/repo/sample_data/rag_evaluation_dataset.csv")
SAMPLE_DIR = Path("/Users/ems/Desktop/ESG_AI/repo/sample_data")
OUTPUT_ROOT = Path("/Users/ems/Desktop/ESG_AI/repo/outputs")
BASE_URL = os.environ.get("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")

api_key = os.environ.get("ALBERT_API_KEY")
if not api_key:
    print("Missing ALBERT_API_KEY. Export it and rerun.")
    sys.exit(1)

companies = []
with DATASET_PATH.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        company = (row.get("company") or "").strip()
        if company and company not in companies:
            companies.append(company)

pdfs = sorted(SAMPLE_DIR.glob("*.pdf"))
if not pdfs:
    print("No PDFs found under sample_data.")
    sys.exit(1)


def pick_pdf(company: str) -> Path:
    best = None
    best_score = -1
    for path in pdfs:
        stem = path.stem.replace("_", " ").replace("-", " ")
        score = max(
            fuzz.token_set_ratio(company.lower(), stem.lower()),
            fuzz.partial_ratio(company.lower(), stem.lower()),
        )
        if score > best_score:
            best_score = score
            best = path
    if best is None:
        raise RuntimeError(f"No PDF match found for {company}")
    return best


client = AlbertClient(api_key=api_key, base_url=BASE_URL)
models = client.list_models()

embed_types = {
    "embedding",
    "embeddings",
    "text-embedding",
    "text-embeddings",
    "text-embeddings-inference",
}
text_types = {"text-generation"}

embedding_models = [m["id"] for m in models if str(m.get("type", "")).lower() in embed_types]
text_models = [m["id"] for m in models if str(m.get("type", "")).lower() in text_types]

embed_override = os.environ.get("EMBEDDING_MODEL_IDS")
if embed_override:
    embedding_models = [item.strip() for item in embed_override.split(",") if item.strip()]

text_override = os.environ.get("TEXT_MODEL_IDS")
if text_override:
    text_models = [item.strip() for item in text_override.split(",") if item.strip()]

if not embedding_models:
    print("No embedding models returned by Albert /models.")
    sys.exit(1)

if not text_models:
    print("No text-generation models returned by Albert /models.")
    sys.exit(1)


def score_to_rating(score: float) -> int:
    if score <= 0:
        return 1
    rating = round(max(1.0, min(5.0, score / 25.0)))
    return int(rating)


def score_answer(answer: str, ground_truth: str, contexts: str) -> float:
    answer = (answer or "").strip().lower()
    ground_truth = (ground_truth or "").strip().lower()
    contexts = (contexts or "").strip().lower()
    if not answer:
        return 0.0

    targets = [ground_truth, contexts]
    scores = []
    for target in targets:
        if not target:
            continue
        scores.append(float(fuzz.partial_ratio(answer, target)))
        scores.append(float(fuzz.token_set_ratio(answer, target)))
    return max(scores) if scores else 0.0


def extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in text_parts if part).strip()
    return ""


def answer_with_prompt(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    text_model: str,
    system_prompt: str,
) -> str:
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            (
                f"{chunk['chunk_id']} | {chunk['source_file']} | pages {chunk['page_start']}-{chunk['page_end']}\n"
                f"{chunk['text']}"
            )
        )
    context = "\n\n".join(context_blocks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]
    response = client._request(
        "POST",
        "/chat/completions",
        json={"model": text_model, "messages": messages, "stream": False},
    ).json()
    return extract_message_content(response)


chunk_grid = []
for target_tokens in (280, 320, 360):
    for min_tokens in (200, 240):
        for max_tokens in (380, 420):
            if min_tokens >= max_tokens or target_tokens > max_tokens:
                continue
            for overlap_tokens in (0, 60, 120):
                chunk_grid.append(
                    {
                        "target_tokens": target_tokens,
                        "min_tokens": min_tokens,
                        "max_tokens": max_tokens,
                        "overlap_tokens": overlap_tokens,
                    }
                )

system_prompts = [
    "You are an ESG analyst. Answer only from the supplied context. If the context is insufficient, say so clearly. Cite supporting chunk ids in brackets.",
    "You are a precise ESG analyst. Use only the provided context. If the answer is not present, say 'Not found in the provided report.' Cite chunk ids in brackets.",
]

embedding_model = embedding_models[0]
rows: list[dict[str, Any]] = []
best_config: dict[str, Any] | None = None
best_score = -1.0

for config in chunk_grid:
    for system_prompt in system_prompts:
        config_scores: list[float] = []
        config_rows: list[dict[str, Any]] = []

        for company in companies:
            pdf_path = pick_pdf(company)
            index_dir = OUTPUT_ROOT / (
                f"rag_index_{company.replace(' ', '_').lower()}_{embedding_model.replace('/', '_')}"
                f"_t{config['target_tokens']}_min{config['min_tokens']}_max{config['max_tokens']}_ov{config['overlap_tokens']}"
            )

            build_index(
                pdf_paths=[pdf_path],
                index_dir=index_dir,
                target_tokens=config["target_tokens"],
                min_tokens=config["min_tokens"],
                max_tokens=config["max_tokens"],
                overlap_tokens=config["overlap_tokens"],
                batch_size=16,
                embedding_model=embedding_model,
                dry_run=False,
                base_url=BASE_URL,
            )

            with DATASET_PATH.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                company_rows = [row for row in reader if (row.get("company") or "").strip() == company]

            for text_model in text_models[:2]:
                scores: list[float] = []
                for row in company_rows:
                    question = row["question"]
                    ground_truth = row.get("ground_truth", "")
                    contexts = row.get("contexts", "")

                    retrieved_chunks, _embedding_model = retrieve_chunks(
                        index_dir=index_dir,
                        question=question,
                        top_k=5,
                        embedding_model=embedding_model,
                        base_url=BASE_URL,
                    )
                    answer = answer_with_prompt(
                        question=question,
                        retrieved_chunks=retrieved_chunks,
                        text_model=text_model,
                        system_prompt=system_prompt,
                    )
                    scores.append(score_answer(answer, ground_truth, contexts))

                avg_score = sum(scores) / len(scores) if scores else 0.0
                rating = score_to_rating(avg_score)
                config_scores.append(avg_score)
                config_rows.append(
                    {
                        "company": company,
                        "text_model": text_model,
                        "pdf": pdf_path.name,
                        "questions": len(company_rows),
                        "avg_score": avg_score,
                        "rating": rating,
                        **config,
                        "system_prompt": system_prompt,
                    }
                )

        overall = sum(config_scores) / len(config_scores) if config_scores else 0.0
        if overall > best_score:
            best_score = overall
            best_config = {
                "config": config,
                "system_prompt": system_prompt,
                "rows": config_rows,
                "overall": overall,
            }

if best_config is None:
    print("No evaluation results were produced.")
    sys.exit(1)

rows = best_config["rows"]
config = best_config["config"]
system_prompt = best_config["system_prompt"]

print("Best configuration:")
print(
    f"- target_tokens: {config['target_tokens']}, min_tokens: {config['min_tokens']}, "
    f"max_tokens: {config['max_tokens']}, overlap_tokens: {config['overlap_tokens']}"
)
print(f"- system_prompt: {system_prompt}")
print(f"- overall_avg_score: {best_config['overall']:.2f}\n")

header = ["Company", "Text Model", "PDF", "Q", "AvgScore", "Rating"]
print("\t".join(header))
for row in rows:
    print(
        "\t".join(
            [
                row["company"],
                row["text_model"],
                row["pdf"],
                str(row["questions"]),
                f"{row['avg_score']:.2f}",
                str(row["rating"]),
            ]
        )
    )

print("\nPer-company average rating (across text models):")
company_scores: dict[str, list[int]] = {}
for row in rows:
    company_scores.setdefault(row["company"], []).append(row["rating"])
for company, scores in company_scores.items():
    avg = sum(scores) / len(scores)
    print(f"- {company}: {avg:.2f}")

print("\nPer-model average rating (across companies):")
model_scores: dict[str, list[int]] = {}
for row in rows:
    model_scores.setdefault(row["text_model"], []).append(row["rating"])
for model, scores in model_scores.items():
    avg = sum(scores) / len(scores)
    print(f"- {model}: {avg:.2f}")
