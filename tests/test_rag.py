import numpy as np

from app.rag import chunk_text, search_vectors


def test_chunk_text_produces_reasonable_chunk_sizes():
    paragraph = "Sustainability strategy and emissions disclosure. " * 75
    text = "\n\n".join([paragraph] * 9)

    chunks = chunk_text(
        text,
        source_file="report.pdf",
        source_path="/tmp/report.pdf",
        target_tokens=360,
        min_tokens=300,
        max_tokens=500,
    )

    assert len(chunks) >= 2
    assert all(250 <= chunk.token_count <= 500 for chunk in chunks)


def test_search_vectors_returns_best_match_first():
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    matches = search_vectors(query, vectors, top_k=2)

    assert matches[0][0] == 0
    assert matches[0][1] >= matches[1][1]
