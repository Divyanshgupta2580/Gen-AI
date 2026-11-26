"""
PDF Ingestion module using PyMuPDF (fitz).
Extracts text page-by-page, computes document fingerprints,
and detects scanned or unreadable documents.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import List
import pymupdf

from utils.validators import (
    validate_pdf_path,
    ValidationError,
    EmptyPDFError,
    CorruptedPDFError,
)


class PDFProcessingError(ValidationError):
    """Base error for PDF processing issues."""
    pass


class ScannedPDFError(PDFProcessingError):
    """Raised when a PDF contains no extractable text (e.g. image-only scans)."""
    pass


@dataclass
class PDFPage:
    """Represents a single page extracted from a PDF document."""
    page_number: int  # 1-indexed
    text: str
    char_count: int


@dataclass
class PDFDocument:
    """Represents an ingested PDF document with metadata and pages."""
    file_path: Path
    document_name: str
    doc_hash: str
    total_pages: int
    file_size_bytes: int
    pages: List[PDFPage] = field(default_factory=list)
    total_characters: int = 0


def calculate_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """
    Calculate SHA-256 fingerprint of a file for caching and deduplication.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_pdf(file_path_str: str) -> PDFDocument:
    """
    Load and extract text page-by-page from a PDF file using PyMuPDF.
    
    Args:
        file_path_str: Absolute or user-provided path to the PDF.
        
    Returns:
        PDFDocument containing page text and document metadata.
        
    Raises:
        InvalidPDFPathError: Path invalid or unreadable.
        EmptyPDFError: Document contains no pages or zero bytes.
        CorruptedPDFError: Document structure is invalid.
        ScannedPDFError: Document has no extractable text (scanned image).
    """
    path = validate_pdf_path(file_path_str)
    file_hash = calculate_file_hash(path)
    file_size = path.stat().st_size
    
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise CorruptedPDFError(f"Failed to open '{path.name}': {e}")
        
    pages: List[PDFPage] = []
    total_chars = 0
    
    try:
        total_pages = doc.page_count
        if total_pages == 0:
            raise EmptyPDFError(f"PDF '{path.name}' contains 0 pages.")
            
        for page_idx in range(total_pages):
            page = doc[page_idx]
            raw_text = page.get_text("text") or ""
            
            # Normalize whitespace while preserving paragraphs
            normalized_text = "\n".join(
                line.strip() for line in raw_text.splitlines() if line.strip()
            )
            
            char_count = len(normalized_text)
            total_chars += char_count
            
            pages.append(
                PDFPage(
                    page_number=page_idx + 1,  # 1-indexed for citations
                    text=normalized_text,
                    char_count=char_count,
                )
            )
    except PDFProcessingError:
        raise
    except Exception as e:
        raise CorruptedPDFError(f"Error reading pages from '{path.name}': {e}")
    finally:
        doc.close()
        
    # Check for scanned image-only PDFs
    # If the total character count across the entire document is negligible (< 50 chars)
    if total_chars < 50:
        raise ScannedPDFError(
            f"The PDF '{path.name}' contains no extractable text (only {total_chars} characters detected across {total_pages} pages). "
            "It appears to be a scanned image-only document. "
            "OCR is not currently supported in this version. "
            "Please provide a text-searchable PDF."
        )
        
    return PDFDocument(
        file_path=path,
        document_name=path.name,
        doc_hash=file_hash,
        total_pages=total_pages,
        file_size_bytes=file_size,
        pages=pages,
        total_characters=total_chars,
    )
