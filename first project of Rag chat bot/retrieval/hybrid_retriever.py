"""
Hybrid Retriever Module.
Coordinates TF-IDF Vector Retrieval, BM25 Keyword Retrieval,
ChromaDB Persistent Vector Storage, and Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import (
    RRF_K,
    TOP_K_TFIDF,
    TOP_K_BM25,
    TOP_K_FINAL,
    CHROMA_PERSIST_DIR,
    CACHE_DIR,
)
from ingestion.chunker import DocumentChunk
from ingestion.pdf_loader import PDFDocument
from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf import reciprocal_rank_fusion
from database.vector_store import ChromaVectorStore


class HybridRetriever:
    """
    Unified Hybrid Retrieval System.
    Combines dense TF-IDF vector similarity with sparse BM25 term weighting,
    persists vectors in ChromaDB, and fuses candidates with Reciprocal Rank Fusion.
    """
    
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.vector_store = ChromaVectorStore(persist_dir or CHROMA_PERSIST_DIR)
        self.tfidf_retriever = TFIDFRetriever(cache_dir or CACHE_DIR)
        self.bm25_retriever = BM25Retriever()
        
        self.active_doc_hash: str = ""
        self.active_collection_name: str = ""
        self.chunks: List[DocumentChunk] = []
        self.is_indexed: bool = False

    def index_document(self, pdf_doc: PDFDocument, chunks: List[DocumentChunk]) -> None:
        """
        Build TF-IDF, BM25, and ChromaDB indexes for the ingested PDF.
        Reuses cached indexes if the document hash matches.
        
        Args:
            pdf_doc: Ingested PDF document.
            chunks: Page-aware chunks extracted from the document.
        """
        self.chunks = chunks
        self.active_doc_hash = pdf_doc.doc_hash
        self.active_collection_name = self.vector_store.get_collection_name(pdf_doc.doc_hash)
        
        # 1. Build or load TF-IDF index
        cached = self.tfidf_retriever.load_cache(pdf_doc.doc_hash)
        if not cached:
            self.tfidf_retriever.fit(chunks, pdf_doc.doc_hash)
            
        # 2. Synchronize with ChromaDB
        if not self.vector_store.collection_exists(self.active_collection_name):
            dense_vectors = self.tfidf_retriever.get_dense_embeddings()
            if dense_vectors:
                self.vector_store.store_chunks(
                    collection_name=self.active_collection_name,
                    chunks=chunks,
                    embeddings=dense_vectors,
                )
                
        # 3. Build BM25 index
        self.bm25_retriever.build_index(chunks, pdf_doc.doc_hash)
        self.is_indexed = True

    def search(
        self,
        query: str,
        top_k_tfidf: int = TOP_K_TFIDF,
        top_k_bm25: int = TOP_K_BM25,
        top_k_final: int = TOP_K_FINAL,
        k_rrf: int = RRF_K,
    ) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Retrieval for a user query.
        
        Workflow:
        1. Query TF-IDF vector index -> top 10 results.
        2. Query BM25 keyword index -> top 10 results.
        3. Merge candidate pools via Reciprocal Rank Fusion (k=60).
        4. Deduplicate and return top 4-5 final chunks.
        
        Args:
            query: User's question or search query.
            top_k_tfidf: Number of vector candidates.
            top_k_bm25: Number of keyword candidates.
            top_k_final: Number of final merged candidates.
            k_rrf: RRF smoothing constant.
            
        Returns:
            List of candidate dictionaries ordered by RRF score.
        """
        if not self.is_indexed or not query or not query.strip():
            return []
            
        # Step 1: Lexical Vector Retrieval (TF-IDF)
        tfidf_results = self.tfidf_retriever.retrieve(query, top_k=top_k_tfidf)
        
        # Step 2: Sparse Keyword Retrieval (BM25)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k_bm25)
        
        # Step 3 & 4: Reciprocal Rank Fusion & Deduplication
        ranked_lists = [
            ("tfidf", tfidf_results),
            ("bm25", bm25_results),
        ]
        
        fused_results = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=k_rrf,
            top_k_final=top_k_final,
        )
        
        return fused_results
