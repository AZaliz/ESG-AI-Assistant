import pytest

from app.rag_eval import _score_retrieval, compute_eval_summary, score_answer_text


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


def test_score_answer_text_marks_supported_answer_as_hit():
    result = score_answer_text(
        "The target is a 63% absolute reduction by 2030.",
        "-63% absolute reduction",
        ["Ambition for 2030 compared to 2015 baseline: CO2 -63%"],
    )

    assert result["status"] == "hit"
    assert result["score"] >= 90


def test_compute_eval_summary_all_hits():
    results = [
        {"status": "hit", "best_match_score": 95.0, "top_score": 0.92},
        {"status": "hit", "best_match_score": 98.0, "top_score": 0.88},
        {"status": "hit", "best_match_score": 91.0, "top_score": 0.85},
    ]

    summary = compute_eval_summary(results)

    assert summary["hit"] == 3
    assert summary["partial"] == 0
    assert summary["miss"] == 0
    assert summary["hit_rate"] == 1.0
    assert summary["partial_or_hit_rate"] == 1.0
    assert summary["average_best_match_score"] == pytest.approx(94.67, abs=0.1)
    assert summary["average_top_retrieval_score"] == pytest.approx(0.8833, abs=0.01)


def test_compute_eval_summary_mixed():
    results = [
        {"status": "hit", "best_match_score": 95.0, "top_score": 0.92},
        {"status": "partial", "best_match_score": 75.0, "top_score": 0.70},
        {"status": "miss", "best_match_score": 45.0, "top_score": 0.50},
        {"status": "hit", "best_match_score": 98.0, "top_score": 0.88},
    ]

    summary = compute_eval_summary(results)

    assert summary["hit"] == 2
    assert summary["partial"] == 1
    assert summary["miss"] == 1
    assert summary["hit_rate"] == 0.5
    assert summary["partial_or_hit_rate"] == 0.75
    assert summary["average_top_retrieval_score"] == pytest.approx(0.75, abs=0.01)


def test_compute_eval_summary_empty():
    summary = compute_eval_summary([])

    assert summary["hit"] == 0
    assert summary["hit_rate"] == 0.0
    assert summary["partial_or_hit_rate"] == 0.0


def test_compute_eval_summary_missing_top_score():
    results = [
        {"status": "hit", "best_match_score": 95.0, "top_score": None},
        {"status": "hit", "best_match_score": 98.0, "top_score": 0.88},
    ]

    summary = compute_eval_summary(results)

    assert summary["average_top_retrieval_score"] == 0.88
