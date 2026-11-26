"""
Tutor Engine for PDF Tutor AI.
Orchestrates PDF Ingestion, Page-Aware Chunking, Hybrid Retrieval (TF-IDF + BM25 + RRF),
ChromaDB persistent vector synchronization, and dual-mode answer generation (Gemini / Retrieval-Only).
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from ingestion.pdf_loader import load_pdf, PDFDocument
from ingestion.chunker import create_chunks, DocumentChunk
from retrieval.hybrid_retriever import HybridRetriever
from llm.gemini_client import GeminiClient, GeminiAPIError
from llm.prompts import build_tutor_prompt, TUTOR_SYSTEM_PROMPT
from tutor.conversation import ConversationManager
from utils.validators import sanitize_input


class TutorEngine:
    """
    Main educational tutoring engine.
    Coordinates document processing, hybrid retrieval, and answer synthesis.
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
    ):
        self.hybrid_retriever = HybridRetriever()
        self.gemini_client = GeminiClient(api_key=gemini_api_key, model_name=gemini_model)
        self.conversation = ConversationManager()
        
        self.active_document: Optional[PDFDocument] = None
        self.active_chunks: List[DocumentChunk] = []

    def set_gemini_api_key(self, api_key: str) -> None:
        """Update Gemini API Key dynamically."""
        self.gemini_client.update_api_key(api_key)

    def set_gemini_model(self, model_name: str) -> None:
        """Update active Gemini model."""
        self.gemini_client.set_model(model_name)

    @property
    def is_gemini_available(self) -> bool:
        """Checks if Gemini API is ready for natural language answer generation."""
        return self.gemini_client.is_available()

    @property
    def is_document_loaded(self) -> bool:
        """Checks if a PDF has been successfully ingested and indexed."""
        return self.active_document is not None and self.hybrid_retriever.is_indexed

    def load_document(self, pdf_path_str: str) -> Dict[str, Any]:
        """
        Ingest, chunk, and index a PDF document.
        
        Args:
            pdf_path_str: Absolute or user-provided path to the PDF.
            
        Returns:
            Dictionary with document metadata and chunk stats.
        """
        # 1. Load document with PyMuPDF
        pdf_doc = load_pdf(pdf_path_str)
        
        # 2. Page-aware chunking with RecursiveCharacterTextSplitter
        chunks = create_chunks(pdf_doc)
        
        # 3. Hybrid Indexing (TF-IDF + ChromaDB + BM25)
        self.hybrid_retriever.index_document(pdf_doc, chunks)
        
        self.active_document = pdf_doc
        self.active_chunks = chunks
        
        # Reset conversation when a new document is loaded
        self.conversation.clear()
        
        return {
            "document_name": pdf_doc.document_name,
            "total_pages": pdf_doc.total_pages,
            "total_chunks": len(chunks),
            "file_size_bytes": pdf_doc.file_size_bytes,
            "doc_hash": pdf_doc.doc_hash,
            "total_characters": pdf_doc.total_characters,
        }

    def answer_query(
        self,
        user_query: str,
        mode: str = "Explain Concept",
        force_retrieval_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Process student query with Hybrid RAG.
        
        Args:
            user_query: Student's question.
            mode: Chosen tutoring mode.
            force_retrieval_only: If True, bypass LLM even if API key is present.
            
        Returns:
            Dictionary with answer, citations, retrieved chunks, and metadata.
        """
        if not self.is_document_loaded:
            raise RuntimeError("No PDF document is currently loaded. Please load a PDF first.")
            
        sanitized_query = sanitize_input(user_query)
        if not sanitized_query:
            return {
                "answer": "Please enter a non-empty question or topic.",
                "citations": [],
                "retrieved_chunks": [],
                "is_retrieval_only": False,
                "mode": mode,
            }
            
        # Record student question in conversation history
        self.conversation.add_user_message(sanitized_query, mode)
        
        # 1. Perform Hybrid Retrieval (TF-IDF + BM25 + RRF)
        retrieved_results = self.hybrid_retriever.search(sanitized_query)
        
        # Extract unique citations
        citations = sorted(
            list(
                set(
                    item["page_number"]
                    for item in retrieved_results
                    if item.get("page_number") is not None
                )
            )
        )
        
        # Decide generation mode
        use_gemini = self.is_gemini_available and not force_retrieval_only
        
        if use_gemini:
            # MODE A: Gemini Answer Generation
            recent_context = self.conversation.get_recent_context(max_turns=2)
            prompt = build_tutor_prompt(
                query=sanitized_query,
                retrieved_passages=retrieved_results,
                mode=mode,
                conversation_history=recent_context,
            )
            
            try:
                answer = self.gemini_client.generate_tutoring_response(
                    prompt=prompt,
                    system_instruction=TUTOR_SYSTEM_PROMPT,
                )
                is_retrieval_only = False
            except GeminiAPIError as e:
                # Graceful fallback to Mode B on API error
                answer = (
                    f"⚠️ **Gemini API Error:** {str(e)}\n\n"
                    "Falling back to **Retrieval-Only Mode**. Below are the relevant passages retrieved from the PDF:"
                )
                is_retrieval_only = True
        else:
            # MODE B: Retrieval-Only Mode
            is_retrieval_only = True
            if not retrieved_results:
                answer = (
                    "ℹ️ **Retrieval-Only Mode** (No Gemini API Key configured)\n\n"
                    "No relevant passages were found in the document matching your query."
                )
            else:
                formatted_lines = [
                    "ℹ️ **Retrieval-Only Mode** (Configure `GEMINI_API_KEY` to enable natural-language tutoring answers)\n",
                    f"Found **{len(retrieved_results)} relevant passage(s)** across pages: {', '.join(f'p. {p}' for p in citations)}\n",
                ]
                for i, r in enumerate(retrieved_results, 1):
                    formatted_lines.append(
                        f"**Passage {i} (Page {r['page_number']} | RRF Score: {r['rrf_score']:.4f})**:\n"
                        f"> {r['text']}\n"
                    )
                answer = "\n".join(formatted_lines)
                
        # Record assistant response
        self.conversation.add_assistant_message(
            content=answer,
            retrieved_chunks=retrieved_results,
            mode=mode,
            is_retrieval_only=is_retrieval_only,
        )
        
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_results,
            "is_retrieval_only": is_retrieval_only,
            "mode": mode,
            "document_name": self.active_document.document_name,
        }
