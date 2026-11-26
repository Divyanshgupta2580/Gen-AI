"""
BM25 Sparse Keyword Retrieval Module.
Uses rank-bm25 (BM25Okapi) for precise keyword, acronym, and technical term matching.
"""

from pathlib import Path
import re
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi

from config.settings import TOP_K_BM25
from ingestion.chunker import DocumentChunk


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into lowercase terms, alphanumeric words, and hyphenated terms.
    Preserves acronyms and technical identifiers (e.g. 'cpu-bound', 'O_RDONLY').
    """
    if not text:
        return []
    # Match alphanumeric words including internal hyphens and underscores
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """
    BM25 retriever using the BM25Okapi probabilistic ranking algorithm.
    Excels at exact keyword, definition, and technical acronym lookups.
    """
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.doc_hash: str = ""

    def build_index(self, chunks: List[DocumentChunk], doc_hash: str) -> None:
        """
        Tokenize document chunks and build the BM25 index.
        
        Args:
            chunks: List of DocumentChunk instances.
            doc_hash: Document fingerprint.
        """
        if not chunks:
            self.bm25 = None
            self.chunks = []
            self.doc_hash = doc_hash
            return
            
        self.chunks = chunks
        self.doc_hash = doc_hash
        
        tokenized_corpus = [tokenize_text(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self, query: str, top_k: int = TOP_K_BM25
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve top-k chunks matching the query using BM25 scoring.
        
        Args:
            query: User's query.
            top_k: Maximum number of chunks to return.
            
        Returns:
            List of (DocumentChunk, bm25_score) tuples ranked by descending score.
        """
        if not query or not query.strip() or self.bm25 is None or not self.chunks:
            return []
            
        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []
            
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        
        results: List[Tuple[DocumentChunk, float]] = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score > 0.0:  # Only return chunks with positive relevance
                results.append((self.chunks[idx], score))
                
        return results
