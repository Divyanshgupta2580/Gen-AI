"""
Tests for Reciprocal Rank Fusion (RRF).
Verifies the RRF formula, duplicate elimination, rank aggregation, and ordering.
"""

from pathlib import Path
import pytest

from ingestion.chunker import DocumentChunk
from retrieval.rrf import reciprocal_rank_fusion


def make_dummy_chunk(chunk_id: str, page_number: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "document_name": "test.pdf",
            "page_number": page_number,
            "chunk_id": chunk_id,
        },
    )


def test_rrf_formula_calculation():
    """
    Test exact Reciprocal Rank Fusion calculation:
    RRF(d) = sum( 1 / (k + rank) )
    For k=60:
    Rank 1 -> 1 / (60 + 1) = 1/61 ≈ 0.01639344
    Rank 2 -> 1 / (60 + 2) = 1/62 ≈ 0.01612903
    """
    c1 = make_dummy_chunk("c1", 1, "text one")
    c2 = make_dummy_chunk("c2", 2, "text two")
    
    ranked_list = [("tfidf", [(c1, 0.9), (c2, 0.5)])]
    results = reciprocal_rank_fusion(ranked_list, k=60, top_k_final=2)
    
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert pytest.approx(results[0]["rrf_score"], rel=1e-5) == 1.0 / 61.0
    assert results[1]["chunk_id"] == "c2"
    assert pytest.approx(results[1]["rrf_score"], rel=1e-5) == 1.0 / 62.0


def test_rrf_duplicate_elimination_and_fusion():
    """
    Test that an item appearing in BOTH rankings gets accumulated score
    and appears ahead of items present in only one ranking.
    """
    c_shared = make_dummy_chunk("c_shared", 1, "Shared important concept")
    c_only_tfidf = make_dummy_chunk("c_tfidf", 2, "Only in tfidf")
    c_only_bm25 = make_dummy_chunk("c_bm25", 3, "Only in bm25")
    
    tfidf_list = [
        (c_shared, 0.9),      # Rank 1 in TF-IDF
        (c_only_tfidf, 0.8),  # Rank 2 in TF-IDF
    ]
    bm25_list = [
        (c_shared, 15.0),     # Rank 1 in BM25
        (c_only_bm25, 10.0),  # Rank 2 in BM25
    ]
    
    ranked_lists = [
        ("tfidf", tfidf_list),
        ("bm25", bm25_list),
    ]
    
    fused = reciprocal_rank_fusion(ranked_lists, k=60, top_k_final=5)
    
    # Must have 3 unique chunks (no duplicate c_shared)
    assert len(fused) == 3
    
    # c_shared must be top rank with combined score: 1/61 + 1/61 = 2/61
    expected_shared_score = (1.0 / 61.0) + (1.0 / 61.0)
    assert fused[0]["chunk_id"] == "c_shared"
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-5) == expected_shared_score
    
    # Check that source metadata was properly stored
    assert "tfidf" in fused[0]["sources"]
    assert "bm25" in fused[0]["sources"]
    assert fused[0]["sources"]["tfidf"]["rank"] == 1
    assert fused[0]["sources"]["bm25"]["rank"] == 1


def test_rrf_empty_inputs():
    """Verify empty input handling."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([("tfidf", [])]) == []
