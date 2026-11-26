"""
Tests for BM25 Keyword Retrieval.
Verifies BM25 index creation, exact keyword lookups, technical acronyms, and empty query safety.
"""

from pathlib import Path
import pytest

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import create_chunks
from retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def indexed_bm25(sample_pdf_path: Path):
    doc = load_pdf(str(sample_pdf_path))
    chunks = create_chunks(doc)
    retriever = BM25Retriever()
    retriever.build_index(chunks, doc.doc_hash)
    return retriever, chunks, doc


def test_bm25_index_construction(indexed_bm25):
    """Verify BM25 index builds properly."""
    retriever, chunks, doc = indexed_bm25
    assert retriever.bm25 is not None
    assert len(retriever.chunks) == len(chunks)


def test_bm25_acronym_and_keyword_retrieval(indexed_bm25):
    """Verify exact lookup of technical term 'PCB' (Process Control Block)."""
    retriever, chunks, doc = indexed_bm25
    results = retriever.retrieve("PCB Process Control Block", top_k=3)
    
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "PCB" in top_chunk.text
    assert top_chunk.page_number == 1
    assert score > 0.0


def test_bm25_phrase_retrieval(indexed_bm25):
    """Verify retrieval of specific concept 'Coffman conditions'."""
    retriever, chunks, doc = indexed_bm25
    results = retriever.retrieve("Coffman conditions mutual exclusion circular wait", top_k=3)
    
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "Coffman" in top_chunk.text
    assert top_chunk.page_number == 2


def test_bm25_empty_query(indexed_bm25):
    """Verify empty or whitespace query handling."""
    retriever, _, _ = indexed_bm25
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
