from app import llm


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, timeout):
        return DummyResponse(self.payload)


def test_list_albert_models_filters_and_sorts(monkeypatch):
    payload = {
        "data": [
            {"type": "embedding", "id": "ignored-embedding"},
            {"type": "text-generation", "id": "z-model"},
            {"type": "text-generation", "id": "a-model"},
        ]
    }
    monkeypatch.setattr(llm, "_albert_session", lambda api_key: DummySession(payload))

    models = llm.list_albert_models("api-key")

    assert [model.model_id for model in models] == ["a-model", "z-model"]
    assert [model.label for model in models] == ["Albert | a-model", "Albert | z-model"]


def test_build_messages_skips_blank_system_prompt():
    messages = llm.build_messages("Hello", "   ")

    assert messages == [{"role": "user", "content": "Hello"}]