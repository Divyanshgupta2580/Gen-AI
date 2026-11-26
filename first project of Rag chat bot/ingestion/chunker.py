"""
Text chunking module using LangChain's RecursiveCharacterTextSplitter.
Implements page-aware chunking to ensure chunk metadata and page citations
never mix across page boundaries.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.pdf_loader import PDFDocument


@dataclass
class DocumentChunk:
    """Represents a text chunk traced strictly to its source PDF page."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    
    @property
    def document_name(self) -> str:
        return self.metadata.get("document_name", "")
        
    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", 0)


def create_chunks(
    pdf_doc: PDFDocument,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Split a PDF document into page-aware chunks using RecursiveCharacterTextSplitter.
    
    Each page is chunked independently to prevent metadata bleeding across pages,
    ensuring every citation points strictly to the true originating page.
    
    Args:
        pdf_doc: Ingested PDFDocument instance.
        chunk_size: Maximum character length per chunk (default: 800).
        chunk_overlap: Overlapping character count (default: 150).
        
    Returns:
        List of DocumentChunk objects with accurate metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=True,
    )
    
    chunks: List[DocumentChunk] = []
    hash_prefix = pdf_doc.doc_hash[:8]
    
    for page in pdf_doc.pages:
        page_text = page.text.strip()
        if not page_text:
            continue
            
        page_splits = splitter.split_text(page_text)
        
        for idx, split_text in enumerate(page_splits):
            clean_split = split_text.strip()
            if not clean_split:
                continue
                
            chunk_id = f"{hash_prefix}_p{page.page_number}_c{idx}"
            metadata = {
                "document_name": pdf_doc.document_name,
                "page_number": page.page_number,
                "chunk_id": chunk_id,
                "doc_hash": pdf_doc.doc_hash,
                "chunk_index": idx,
                "char_length": len(clean_split),
            }
            
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=clean_split,
                    metadata=metadata,
                )
            )
            
    return chunks
