"""
Tests for TF-IDF Vector Retrieval.
Verifies vectorization, cosine similarity ranking, empty query handling,
caching, and dense embedding export.
"""

from pathlib import Path
import pytest

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import create_chunks
from retrieval.tfidf_retriever import TFIDFRetriever


@pytest.fixture
def indexed_tfidf(sample_pdf_path: Path, tmp_path: Path):
    doc = load_pdf(str(sample_pdf_path))
    chunks = create_chunks(doc)
    retriever = TFIDFRetriever(cache_dir=tmp_path / "cache")
    retriever.fit(chunks, doc.doc_hash)
    return retriever, chunks, doc


def test_tfidf_vector_generation(indexed_tfidf):
    """Test TF-IDF fitting, vocabulary generation, and dense embeddings."""
    retriever, chunks, doc = indexed_tfidf
    assert retriever.vectorizer is not None
    assert retriever.tfidf_matrix is not None
    assert retriever.tfidf_matrix.shape[0] == len(chunks)
    assert retriever.tfidf_matrix.shape[1] > 10  # Rich vocabulary
    
    # Test dense embeddings for ChromaDB
    dense = retriever.get_dense_embeddings()
    assert len(dense) == len(chunks)
    assert len(dense[0]) == retriever.tfidf_matrix.shape[1]


def test_tfidf_similarity_ranking(indexed_tfidf):
    """Test that a query for 'deadlock' retrieves Deadlock passages at the top."""
    retriever, chunks, doc = indexed_tfidf
    results = retriever.retrieve("What is a deadlock and four Coffman conditions?", top_k=5)
    
    assert len(results) > 0
    top_chunk, top_score = results[0]
    assert "deadlock" in top_chunk.text.lower()
    assert top_chunk.page_number == 2
    assert top_score > 0.0


def test_tfidf_empty_query(indexed_tfidf):
    """Test that an empty query returns an empty list without errors."""
    retriever, _, _ = indexed_tfidf
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_tfidf_out_of_vocabulary(indexed_tfidf):
    """Test that queries with out-of-vocabulary words return empty list without crash."""
    retriever, _, _ = indexed_tfidf
    results = retriever.retrieve("xyzzyqwerasdf1234")
    assert results == []


def test_tfidf_caching(sample_pdf_path: Path, tmp_path: Path):
    """Verify serialization and deserialization of fitted TF-IDF indexes."""
    doc = load_pdf(str(sample_pdf_path))
    chunks = create_chunks(doc)
    cache_dir = tmp_path / "cache"
    
    retriever1 = TFIDFRetriever(cache_dir=cache_dir)
    retriever1.fit(chunks, doc.doc_hash)
    
    # Create second retriever and load from cache
    retriever2 = TFIDFRetriever(cache_dir=cache_dir)
    loaded = retriever2.load_cache(doc.doc_hash)
    assert loaded is True
    assert retriever2.vectorizer is not None
    assert retriever2.tfidf_matrix.shape == retriever1.tfidf_matrix.shape
    
    # Both should yield identical search results
    res1 = retriever1.retrieve("paging memory fault")
    res2 = retriever2.retrieve("paging memory fault")
    assert len(res1) == len(res2)
    assert res1[0][0].chunk_id == res2[0][0].chunk_id
