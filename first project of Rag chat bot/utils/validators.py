"""
Validation and security utilities for PDF Tutor AI.
Provides strict file validation, path checking, and input sanitization.
"""

from pathlib import Path
import re
import pymupdf


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class InvalidPDFPathError(ValidationError):
    """Raised when the PDF path is invalid or non-existent."""
    pass


class CorruptedPDFError(ValidationError):
    """Raised when a PDF file is corrupted or unreadable."""
    pass


class EmptyPDFError(ValidationError):
    """Raised when a PDF file has zero size or zero pages."""
    pass


class EncryptedPDFError(ValidationError):
    """Raised when a PDF is password protected."""
    pass


def validate_pdf_path(path_str: str) -> Path:
    """
    Validate that an absolute or user-provided path points to an accessible,
    valid, unencrypted, and non-empty PDF file.
    
    Args:
        path_str: File path provided by the user.
        
    Returns:
        Resolved absolute Path object.
        
    Raises:
        InvalidPDFPathError: Path does not exist or is not a PDF.
        EmptyPDFError: File has zero bytes or zero pages.
        CorruptedPDFError: File is damaged and cannot be parsed.
        EncryptedPDFError: File requires a password.
    """
    if not path_str or not isinstance(path_str, str) or not path_str.strip():
        raise InvalidPDFPathError("Please enter a valid PDF file path.")
    
    clean_path_str = path_str.strip()
    path = Path(clean_path_str).expanduser()
    
    # Resolve path
    try:
        resolved_path = path.resolve()
    except Exception as e:
        raise InvalidPDFPathError(f"Could not resolve file path: {e}")
        
    if not resolved_path.exists():
        raise InvalidPDFPathError(f"File not found: '{resolved_path}'. Please check that the file path is correct.")
        
    if not resolved_path.is_file():
        raise InvalidPDFPathError(f"Path points to a directory, not a file: '{resolved_path}'.")
        
    if resolved_path.suffix.lower() != ".pdf":
        raise InvalidPDFPathError(
            f"Invalid file type: '{resolved_path.name}'. Only PDF (.pdf) files are supported."
        )
        
    # Check file size
    try:
        file_size = resolved_path.stat().st_size
    except Exception as e:
        raise InvalidPDFPathError(f"Failed to access file metadata: {e}")
        
    if file_size == 0:
        raise EmptyPDFError(f"The PDF file '{resolved_path.name}' is empty (0 bytes).")
        
    # Verify with PyMuPDF
    try:
        doc = pymupdf.open(resolved_path)
    except Exception as e:
        raise CorruptedPDFError(f"The PDF file '{resolved_path.name}' is corrupted or unreadable: {e}")
        
    try:
        if doc.is_encrypted:
            raise EncryptedPDFError(
                f"The PDF '{resolved_path.name}' is encrypted or password-protected. "
                "Please provide an unencrypted PDF."
            )
            
        if doc.page_count == 0:
            raise EmptyPDFError(f"The PDF '{resolved_path.name}' contains 0 pages.")
    finally:
        doc.close()
        
    return resolved_path


def sanitize_input(text: str, max_length: int = 4000) -> str:
    """
    Sanitize user queries to prevent prompt injection and remove control characters.
    
    Args:
        text: Raw user text.
        max_length: Maximum allowed characters.
        
    Returns:
        Cleaned string.
    """
    if not text:
        return ""
    # Strip null bytes and non-printable control characters (except newline, tab)
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    sanitized = sanitized.strip()
    return sanitized[:max_length]


def sanitize_collection_name(name: str) -> str:
    """
    ChromaDB collection names must be 3-63 characters, start and end with an
    alphanumeric character, and contain only alphanumeric, underscores, or hyphens.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    if len(cleaned) < 3:
        cleaned = f"pdf_{cleaned}".ljust(3, "0")
    return cleaned[:63]
