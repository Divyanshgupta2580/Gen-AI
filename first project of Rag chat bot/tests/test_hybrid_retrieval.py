"""
Tests for End-to-End Hybrid Retrieval (TF-IDF + BM25 + ChromaDB + RRF).
"""

from pathlib import Path
import pytest

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import create_chunks
from retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def hybrid_setup(sample_pdf_path: Path, tmp_path: Path):
    doc = load_pdf(str(sample_pdf_path))
    chunks = create_chunks(doc)
    
    persist_dir = tmp_path / "chroma_db"
    cache_dir = tmp_path / "cache"
    
    retriever = HybridRetriever(persist_dir=persist_dir, cache_dir=cache_dir)
    retriever.index_document(doc, chunks)
    return retriever, doc, chunks


def test_chromadb_storage_and_synchronization(hybrid_setup):
    """Verify ChromaDB stores chunks and vector representations accurately."""
    retriever, doc, chunks = hybrid_setup
    collection_name = retriever.active_collection_name
    
    assert retriever.vector_store.collection_exists(collection_name)
    count = retriever.vector_store.get_chunk_count(collection_name)
    assert count == len(chunks)


def test_hybrid_retrieval_query_and_fused_ranking(hybrid_setup):
    """
    Test hybrid search across technical concepts.
    Verify top results match relevant page numbers and include both source ranks.
    """
    retriever, doc, chunks = hybrid_setup
    
    # Query for Virtual Memory and Paging (located on Page 3)
    results = retriever.search("How does paging and page fault work in virtual memory?", top_k_final=4)
    
    assert len(results) > 0
    assert len(results) <= 4
    top_result = results[0]
    
    assert top_result["page_number"] == 3
    assert "page" in top_result["text"].lower()
    assert top_result["rrf_score"] > 0.0
    assert "sources" in top_result


def test_hybrid_retrieval_deadlocks(hybrid_setup):
    """Test retrieval for deadlock concepts (located on Page 2)."""
    retriever, doc, chunks = hybrid_setup
    results = retriever.search("deadlock Coffman conditions hold and wait", top_k_final=5)
    
    assert len(results) > 0
    top_pages = [r["page_number"] for r in results]
    assert 2 in top_pages  # Page 2 must be represented
