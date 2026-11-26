"""
Tests for page-aware text chunking.
Verifies chunk sizing, overlap, page metadata integrity, and page isolation.
"""

from pathlib import Path
import pytest

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import create_chunks, DocumentChunk


def test_chunking_creation_and_metadata(sample_pdf_path: Path):
    """Verify that chunks are created and each preserves source page metadata."""
    doc = load_pdf(str(sample_pdf_path))
    chunks = create_chunks(doc, chunk_size=300, chunk_overlap=50)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.chunk_id != ""
        assert chunk.text != ""
        assert chunk.page_number in [1, 2, 3]
        assert chunk.document_name == "OperatingSystems.pdf"
        assert "doc_hash" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_page_aware_isolation(sample_pdf_path: Path):
    """
    Ensure page boundaries are respected: text from Page 1 should not
    bleed into a chunk labelled with Page 2.
    """
    doc = load_pdf(str(sample_pdf_path))
    chunks = create_chunks(doc, chunk_size=200, chunk_overlap=40)
    
    page_1_chunks = [c for c in chunks if c.page_number == 1]
    page_2_chunks = [c for c in chunks if c.page_number == 2]
    page_3_chunks = [c for c in chunks if c.page_number == 3]
    
    assert len(page_1_chunks) > 0
    assert len(page_2_chunks) > 0
    assert len(page_3_chunks) > 0
    
    # Process Management is only on Page 1
    for c in page_1_chunks:
        assert "Virtual Memory" not in c.text
        assert "Coffman" not in c.text
        
    # Deadlocks are only on Page 2
    for c in page_2_chunks:
        assert "Process Control Block" not in c.text
        assert "page fault" not in c.text


def test_chunk_size_adherence(sample_pdf_path: Path):
    """Verify that chunk character lengths adhere to configured size bounds."""
    doc = load_pdf(str(sample_pdf_path))
    max_size = 400
    overlap = 80
    chunks = create_chunks(doc, chunk_size=max_size, chunk_overlap=overlap)
    
    for chunk in chunks:
        # Chunks should not substantially exceed max_size (small leeway for separator boundary)
        assert len(chunk.text) <= max_size + 50
