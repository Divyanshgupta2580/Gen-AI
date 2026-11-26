"""
Tests for PDF ingestion and validation.
Verifies valid PDFs, invalid paths, empty files, corrupted data, and scanned document detection.
"""

from pathlib import Path
import pytest

from ingestion.pdf_loader import load_pdf, ScannedPDFError
from utils.validators import (
    validate_pdf_path,
    InvalidPDFPathError,
    EmptyPDFError,
    CorruptedPDFError,
)


def test_valid_pdf_loading(sample_pdf_path: Path):
    """Test loading a valid, text-rich PDF document."""
    doc = load_pdf(str(sample_pdf_path))
    assert doc.document_name == "OperatingSystems.pdf"
    assert doc.total_pages == 3
    assert len(doc.pages) == 3
    assert doc.total_characters > 100
    assert doc.doc_hash != ""
    assert doc.pages[0].page_number == 1
    assert "Process Management" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Deadlocks" in doc.pages[1].text
    assert doc.pages[2].page_number == 3
    assert "Virtual Memory" in doc.pages[2].text


def test_invalid_path_nonexistent():
    """Test that a non-existent path raises InvalidPDFPathError."""
    with pytest.raises(InvalidPDFPathError) as exc_info:
        load_pdf("/non/existent/path/document.pdf")
    assert "File not found" in str(exc_info.value)


def test_directory_path(tmp_path: Path):
    """Test that providing a directory path raises InvalidPDFPathError."""
    with pytest.raises(InvalidPDFPathError) as exc_info:
        load_pdf(str(tmp_path))
    assert "points to a directory" in str(exc_info.value)


def test_non_pdf_extension(tmp_path: Path):
    """Test that providing a text file raises InvalidPDFPathError."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Some text notes.")
    with pytest.raises(InvalidPDFPathError) as exc_info:
        load_pdf(str(txt_file))
    assert "Only PDF" in str(exc_info.value)


def test_empty_pdf(empty_pdf_path: Path):
    """Test that a 0-byte file raises EmptyPDFError."""
    with pytest.raises(EmptyPDFError) as exc_info:
        load_pdf(str(empty_pdf_path))
    assert "empty" in str(exc_info.value)


def test_corrupted_pdf(corrupted_pdf_path: Path):
    """Test that a corrupted PDF raises CorruptedPDFError."""
    with pytest.raises(CorruptedPDFError) as exc_info:
        load_pdf(str(corrupted_pdf_path))
    assert "corrupted" in str(exc_info.value) or "Failed to open" in str(exc_info.value)


def test_scanned_pdf_detection(scanned_pdf_path: Path):
    """Test that a scanned/image-only PDF with zero text triggers ScannedPDFError with OCR notice."""
    with pytest.raises(ScannedPDFError) as exc_info:
        load_pdf(str(scanned_pdf_path))
    assert "scanned image-only" in str(exc_info.value)
    assert "OCR is not currently supported" in str(exc_info.value)
