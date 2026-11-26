"""
TF-IDF Vector Retrieval Module.

NOTE ON VECTORIZATION ARCHITECTURE:
This module implements lightweight lexical vector retrieval using scikit-learn's
TfidfVectorizer and cosine similarity. It operates strictly as a statistical
and lexical vector model based on n-gram term frequencies and inverse document
frequencies. It does NOT use neural networks, transformer models, or semantic
embeddings. It runs 100% locally on CPU with zero neural model overhead.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import CACHE_DIR, TOP_K_TFIDF
from ingestion.chunker import DocumentChunk


class TFIDFRetriever:
    """
    Lexical vector retriever using TF-IDF and Cosine Similarity.
    Supports serialization and synchronization with persistent storage.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.chunks: List[DocumentChunk] = []
        self.doc_hash: str = ""

    def _get_cache_path(self, doc_hash: str) -> Path:
        return self.cache_dir / f"{doc_hash}_tfidf.joblib"

    def fit(self, chunks: List[DocumentChunk], doc_hash: str) -> None:
        """
        Fit the TF-IDF vectorizer on chunk texts and persist index.
        
        Args:
            chunks: List of DocumentChunk instances.
            doc_hash: Unique fingerprint of the parent PDF.
        """
        if not chunks:
            self.chunks = []
            self.vectorizer = None
            self.tfidf_matrix = None
            self.doc_hash = doc_hash
            return
            
        self.chunks = chunks
        self.doc_hash = doc_hash
        
        # Configure lexical n-gram vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            lowercase=True,
        )
        
        corpus = [chunk.text for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # Persist index to disk
        self._save_cache(doc_hash)

    def _save_cache(self, doc_hash: str) -> None:
        """Serialize fitted vectorizer and index."""
        cache_file = self._get_cache_path(doc_hash)
        payload = {
            "doc_hash": doc_hash,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "chunks": self.chunks,
        }
        joblib.dump(payload, cache_file)

    def load_cache(self, doc_hash: str) -> bool:
        """
        Load cached index for an unchanged PDF.
        
        Returns:
            True if cache loaded successfully, False otherwise.
        """
        cache_file = self._get_cache_path(doc_hash)
        if not cache_file.exists():
            return False
            
        try:
            payload = joblib.load(cache_file)
            if payload.get("doc_hash") == doc_hash and payload.get("vectorizer") is not None:
                self.doc_hash = doc_hash
                self.vectorizer = payload["vectorizer"]
                self.tfidf_matrix = payload["tfidf_matrix"]
                self.chunks = payload["chunks"]
                return True
        except Exception:
            return False
        return False

    def get_dense_embeddings(self) -> List[List[float]]:
        """
        Convert TF-IDF sparse matrix to dense float vectors for ChromaDB storage.
        
        Returns:
            List of float vectors, one per chunk.
        """
        if self.tfidf_matrix is None:
            return []
        dense_array = self.tfidf_matrix.toarray().astype(float)
        return dense_array.tolist()

    def transform_query(self, query: str) -> List[float]:
        """Transform a single query string into a dense vector."""
        if self.vectorizer is None or not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        return q_vec.toarray()[0].astype(float).tolist()

    def retrieve(
        self, query: str, top_k: int = TOP_K_TFIDF
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the top-k most relevant chunks using cosine similarity over TF-IDF vectors.
        
        Args:
            query: User's question or search query.
            top_k: Number of chunks to retrieve.
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples ranked by descending score.
        """
        if not query or not query.strip() or self.vectorizer is None or self.tfidf_matrix is None:
            return []
            
        if not self.chunks:
            return []
            
        query_vec = self.vectorizer.transform([query])
        
        # Check if query vector is entirely zero (e.g. out of vocabulary terms)
        if query_vec.nnz == 0:
            return []
            
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.0:  # Only return passages with non-zero similarity
                results.append((self.chunks[idx], score))
                
        return results
