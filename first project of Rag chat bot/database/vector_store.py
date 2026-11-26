"""
ChromaDB Persistent Vector Store for PDF Tutor AI.
Stores lexical vector representations and associated document metadata.
Explicitly configured with embedding_function=None to prevent any neural
model downloads or Hugging Face downloads.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config.settings import CHROMA_PERSIST_DIR
from ingestion.chunker import DocumentChunk
from utils.validators import sanitize_collection_name


class ChromaVectorStore:
    """
    Manages persistent storage and querying of vector representations in ChromaDB.
    Guarantees no local neural networks or external downloads by taking
    pre-computed dense vector representations directly.
    """
    
    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure ChromaDB with telemetry disabled and persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

    def get_collection_name(self, doc_hash: str) -> str:
        """Derive a safe collection name from document hash."""
        return sanitize_collection_name(f"pdf_{doc_hash[:16]}")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection already exists and has records."""
        try:
            coll = self.client.get_collection(name=collection_name, embedding_function=None)
            return coll.count() > 0
        except Exception:
            return False

    def store_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """
        Store chunks and their vector representations in ChromaDB.
        
        Args:
            collection_name: Unique collection identifier.
            chunks: List of DocumentChunk objects.
            embeddings: Dense float vectors (e.g. TF-IDF vectors).
        """
        if not chunks:
            return
            
        # Get or create collection with cosine similarity and NO neural embedding function
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        
        # Prepare batch data
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to ChromaDB (upsert if existing)
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query the persistent vector store using a pre-computed vector.
        
        Args:
            collection_name: Target collection.
            query_embedding: Dense float vector of the query.
            top_k: Number of neighbors to return.
            
        Returns:
            List of result dictionaries containing id, text, metadata, and distance.
        """
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None,
            )
        except Exception:
            return []
            
        count = collection.count()
        if count == 0:
            return []
            
        n_results = min(top_k, count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        output = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            
            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity = max(0.0, 1.0 - dist)
                output.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": similarity,
                })
                
        return output

    def get_chunk_count(self, collection_name: str) -> int:
        """Return the count of items in the collection."""
        try:
            coll = self.client.get_collection(name=collection_name, embedding_function=None)
            return coll.count()
        except Exception:
            return 0
