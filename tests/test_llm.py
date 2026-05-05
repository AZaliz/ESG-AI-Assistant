from app.llm import ChatModel, DEFAULT_FREE_MODELS, suggest_free_fallbacks


def test_suggest_free_fallbacks_when_only_one_model_exists():
    models = [ChatModel(provider="albert", model_id="model-a", label="Albert | model-a")]

    suggestions = suggest_free_fallbacks(models)

    assert [item.model_id for item in suggestions] == [model_id for model_id, _label in DEFAULT_FREE_MODELS]
    assert all(item.provider == "ollama" for item in suggestions)
    assert all(not item.available for item in suggestions)


def test_suggest_free_fallbacks_not_added_when_multiple_models_exist():
    models = [
        ChatModel(provider="albert", model_id="model-a", label="Albert | model-a"),
        ChatModel(provider="ollama", model_id="qwen2.5:7b", label="Ollama | qwen2.5:7b"),
    ]

    assert suggest_free_fallbacks(models) == []