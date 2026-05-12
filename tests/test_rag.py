import numpy as np

from app.rag import (
    ChunkRecord,
    _normalize_retrieval_mode,
    _starts_with_header,
    chunk_text,
    search_vectors,
    select_chunk_matches,
)


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


def test_chunk_text_with_overlap():
    paragraph = "Sustainability strategy and emissions disclosure. " * 40
    text = "\n\n".join([f"Section {i}. This is the content for section {i}. " + paragraph for i in range(1, 6)])

    chunks_no_overlap = chunk_text(
        text,
        source_file="report.pdf",
        source_path="/tmp/report.pdf",
        target_tokens=200,
        min_tokens=100,
        max_tokens=400,
        overlap_tokens=0,
    )
    chunks_with_overlap = chunk_text(
        text,
        source_file="report.pdf",
        source_path="/tmp/report.pdf",
        target_tokens=200,
        min_tokens=100,
        max_tokens=400,
        overlap_tokens=80,
    )

    assert len(chunks_no_overlap) >= 1
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_chunk_text_section_aware():
    paragraph = "Detailed sustainability disclosure content. " * 40
    sections = [
        "1. INTRODUCTION",
        "Introduction content. " + paragraph,
        "2. EMISSIONS TARGETS",
        "Emissions targets content. " + paragraph,
        "3. GOVERNANCE FRAMEWORK",
        "Governance content. " + paragraph,
    ]
    text = "\n\n".join(sections)

    chunks_section = chunk_text(
        text,
        source_file="report.pdf",
        source_path="/tmp/report.pdf",
        target_tokens=200,
        min_tokens=80,
        max_tokens=400,
        overlap_tokens=0,
        section_aware=True,
    )
    chunks_plain = chunk_text(
        text,
        source_file="report.pdf",
        source_path="/tmp/report.pdf",
        target_tokens=200,
        min_tokens=80,
        max_tokens=400,
        overlap_tokens=0,
        section_aware=False,
    )

    assert len(chunks_section) >= 3
    assert len(chunks_section) >= len(chunks_plain)


def test_starts_with_header():
    assert _starts_with_header("1. Introduction")
    assert _starts_with_header("2.1 EMISSIONS TARGETS")
    assert _starts_with_header("III. Methodology")
    assert _starts_with_header("GOVERNANCE FRAMEWORK")
    assert not _starts_with_header("This is a regular paragraph with lots of text that goes on and on.")
    assert not _starts_with_header("")


def test_normalize_retrieval_mode():
    assert _normalize_retrieval_mode("dense") == "semantic"
    assert _normalize_retrieval_mode("semantic") == "semantic"
    assert _normalize_retrieval_mode("lexical") == "lexical"
    assert _normalize_retrieval_mode("hybrid") == "hybrid"
    assert _normalize_retrieval_mode("semantic_rerank") == "semantic_rerank"

    try:
        _normalize_retrieval_mode("invalid")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


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


def test_select_chunk_matches_dense_returns_semantic_only():
    chunks = [
        ChunkRecord("chunk-0001", "report.pdf", "/tmp/report.pdf", 1, 1, 12, "General sustainability overview."),
        ChunkRecord(
            "chunk-0002",
            "report.pdf",
            "/tmp/report.pdf",
            2,
            2,
            18,
            "The Scope 1 emissions target for 2030 is a 63 percent reduction from the baseline.",
        ),
        ChunkRecord("chunk-0003", "report.pdf", "/tmp/report.pdf", 3, 3, 14, "Employee volunteering increased year over year."),
    ]

    matches = select_chunk_matches(
        question="What is the Scope 1 emissions target for 2030?",
        chunks=chunks,
        semantic_matches=[(0, 0.95), (1, 0.72), (2, 0.61)],
        top_k=2,
        retrieval_architecture="dense",
        search_breadth=3,
    )

    assert len(matches) == 2
    assert matches[0][0] == 0


def test_select_chunk_matches_lexical_uses_bm25():
    chunks = [
        ChunkRecord("chunk-0001", "report.pdf", "/tmp/report.pdf", 1, 1, 12, "General sustainability overview of the company."),
        ChunkRecord(
            "chunk-0002",
            "report.pdf",
            "/tmp/report.pdf",
            2,
            2,
            18,
            "The Scope 1 emissions target for 2030 is a 63 percent reduction from the baseline.",
        ),
        ChunkRecord("chunk-0003", "report.pdf", "/tmp/report.pdf", 3, 3, 14, "Employee volunteering increased year over year."),
    ]

    matches = select_chunk_matches(
        question="Scope 1 emissions target 2030 reduction",
        chunks=chunks,
        semantic_matches=[(0, 0.95), (1, 0.72), (2, 0.61)],
        top_k=2,
        retrieval_architecture="lexical",
        search_breadth=3,
    )

    assert len(matches) >= 1
    assert matches[0][0] == 1


def test_select_chunk_matches_hybrid_promotes_lexically_supported_chunk():
    chunks = [
        ChunkRecord("chunk-0001", "report.pdf", "/tmp/report.pdf", 1, 1, 12, "General sustainability overview."),
        ChunkRecord(
            "chunk-0002",
            "report.pdf",
            "/tmp/report.pdf",
            2,
            2,
            18,
            "The Scope 1 emissions target for 2030 is a 63 percent reduction from the baseline.",
        ),
        ChunkRecord("chunk-0003", "report.pdf", "/tmp/report.pdf", 3, 3, 14, "Employee volunteering increased year over year."),
    ]

    matches = select_chunk_matches(
        question="What is the Scope 1 emissions target for 2030?",
        chunks=chunks,
        semantic_matches=[(0, 0.95), (1, 0.72), (2, 0.61)],
        top_k=2,
        retrieval_architecture="hybrid",
        search_breadth=3,
    )

    assert matches[0][0] == 1


def test_select_chunk_matches_hybrid_rrf_combines_rankings():
    chunks = [
        ChunkRecord("chunk-0001", "report.pdf", "/tmp/report.pdf", 1, 1, 15, "emissions scope 1 reduction target of 63 percent."),
        ChunkRecord("chunk-0002", "report.pdf", "/tmp/report.pdf", 2, 2, 12, "General company overview and history."),
        ChunkRecord("chunk-0003", "report.pdf", "/tmp/report.pdf", 3, 3, 14, "Employee volunteering and community engagement."),
    ]

    matches = select_chunk_matches(
        question="scope 1 emissions reduction target",
        chunks=chunks,
        semantic_matches=[(2, 0.60), (0, 0.55), (1, 0.50)],
        top_k=3,
        retrieval_architecture="hybrid",
        search_breadth=3,
    )

    assert len(matches) == 3
    assert matches[0][0] == 0
