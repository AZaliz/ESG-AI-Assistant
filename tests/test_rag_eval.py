from app.rag_eval import _score_retrieval


def test_score_retrieval_marks_exact_support_as_hit():
    result = _score_retrieval(
        ["overall reduction of 160 kt CO2e/year"],
        [{"chunk_id": "chunk-0001", "text": "They will enable an overall reduction of 160 kt CO2e/year."}],
    )

    assert result["status"] == "hit"
    assert result["best_chunk_id"] == "chunk-0001"


def test_score_retrieval_marks_unrelated_context_as_miss():
    result = _score_retrieval(
        ["taxonomy aligned capex of 42 percent"],
        [{"chunk_id": "chunk-0004", "text": "Methane emissions from operated sites fell sharply versus 2020."}],
    )

    assert result["status"] == "miss"
