"""
Pytest fixtures for PDF Tutor AI tests.
Generates programmatic sample PDFs for testing edge cases without external files.
"""

import pytest
from pathlib import Path
import pymupdf


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a valid multi-page PDF document with realistic educational text."""
    pdf_path = tmp_path / "OperatingSystems.pdf"
    doc = pymupdf.open()
    
    # Page 1: Introduction to Operating Systems & Processes
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Operating Systems: Architecture and Design\n\n"
        "Chapter 1: Process Management\n"
        "A process is an instance of a computer program that is being executed by one or many threads.\n"
        "It contains the program code and its current activity. Depending on the operating system,\n"
        "a process may be made up of multiple threads of execution that execute instructions concurrently.\n"
        "The Process Control Block (PCB) is a data structure maintained by the operating system for every process.\n"
        "The PCB stores process ID, program counter, CPU registers, CPU scheduling information, and memory management information."
    )
    
    # Page 2: Deadlocks and Concurrency
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "Chapter 2: Concurrency and Deadlocks\n"
        "A deadlock is a situation in which two or more competing actions are each waiting for the other to finish,\n"
        "and thus neither ever does. In an operating system, a deadlock occurs when a process enters a waiting state\n"
        "because a requested system resource is held by another waiting process.\n\n"
        "The Coffman conditions are the four necessary and sufficient conditions for deadlock:\n"
        "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode.\n"
        "2. Hold and Wait: A process must be concurrently holding at least one resource and requesting additional resources.\n"
        "3. No Preemption: Resources cannot be preempted; a resource can only be released voluntarily.\n"
        "4. Circular Wait: A closed chain of processes exists, where each process holds resources needed by the next."
    )
    
    # Page 3: Virtual Memory and Paging
    page3 = doc.new_page()
    page3.insert_text(
        (50, 72),
        "Chapter 3: Virtual Memory and Paging\n"
        "Virtual memory is a memory management capability of an operating system that uses hardware and software\n"
        "to allow a computer to address more memory than is physically installed.\n"
        "Paging is a memory management scheme by which an OS stores and retrieves data from secondary storage for use in main memory.\n"
        "The page table maps virtual addresses to physical frame addresses. When a requested page is not in main memory,\n"
        "a page fault occurs, prompting the OS to retrieve it from disk."
    )
    
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf_path(tmp_path: Path) -> Path:
    """Create a 0-byte file with .pdf extension."""
    path = tmp_path / "empty.pdf"
    path.touch()
    return path


@pytest.fixture
def corrupted_pdf_path(tmp_path: Path) -> Path:
    """Create a corrupted PDF file containing invalid binary data."""
    path = tmp_path / "corrupted.pdf"
    path.write_bytes(b"%PDF-1.4\nCorrupted random byte stream\x00\xff\xfe\xca\xfe")
    return path


@pytest.fixture
def scanned_pdf_path(tmp_path: Path) -> Path:
    """Create a PDF with blank pages (simulating scanned image-only PDF with zero text)."""
    pdf_path = tmp_path / "scanned_invoice.pdf"
    doc = pymupdf.open()
    doc.new_page()  # Blank page without text
    doc.new_page()  # Another blank page
    doc.save(pdf_path)
    doc.close()
    return pdf_path
