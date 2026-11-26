"""
RAG Engine: Integrates Vector Retrieval with Semantic Context Synthesis
Answers user questions using the ingested document chunks stored in ChromaDB.
"""

from typing import Dict, Any, List
import re
from src.embedder import VectorEmbedder
from src.database import VectorDatabase
from src.models import SearchResult


class RAGEngine:
    """
    Retrieval-Augmented Generation query engine.
    Finds top matching chunks and synthesizes an accurate, grounded answer with citations.
    """

    def __init__(self, persist_dir: str = "./vector_store"):
        self.persist_dir = persist_dir
        self.embedder = VectorEmbedder()
        self.db = VectorDatabase(persist_dir=persist_dir)

    def reload(self):
        """Reload database connection to pick up newly indexed chunks."""
        self.db = VectorDatabase(persist_dir=self.persist_dir)

    def answer_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves relevant document chunks and synthesizes a grounded response.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "query": query,
                "answer": "Please enter a question about Multiverse or your uploaded document.",
                "sources": [],
                "confidence": 0.0,
            }

        # Friendly conversational greeting handler
        lower_q = clean_query.lower()
        if lower_q in {"hi", "hello", "hey", "greetings", "who are you", "what can you do", "help"}:
            return {
                "query": query,
                "answer": (
                    "Greetings! I am the **Multiverse™ RAG Intelligence Chatbot**, fully connected to the "
                    "multidimensional knowledge base. You can ask me anything about the **Multiverse Reality Engine**, "
                    "including:\n\n"
                    "- 🌌 **Quantum Singularity Core & 128 PFLOPS TPU**\n"
                    "- 🌀 **Einstein-Rosen Wormhole Interconnects & <0.04ps Latency**\n"
                    "- ⚡ **Parallel Universe Simulations (16,384 concurrent timelines)**\n"
                    "- 🛡️ **Sub-space Security & Timeline Isolation Protocols**\n"
                    "- 🔋 **Zero-Point Quantum Flux Lattice & Monolithic Hardware**\n"
                    "- 💻 **Multiverse OS v4.2 & Python SDK (`import multiverse`)**\n\n"
                    "What would you like to explore across the Multiverse?"
                ),
                "top_section": "Executive Overview > 1.1 Product Mission",
                "confidence": 1.0,
                "sources": [],
            }

        # Retrieve top-k chunks from vector database
        results: List[SearchResult] = self.db.search(
            query_text=clean_query,
            embedder=self.embedder,
            n_results=top_k,
        )

        if not results or results[0].similarity_score < 0.15:
            return {
                "query": query,
                "answer": (
                    f"I could not find sufficiently relevant information in the currently ingested document "
                    f"to answer: \"{query}\". Try rephrasing your question or verify the loaded document content."
                ),
                "sources": [],
                "confidence": 0.0,
            }

        best_result = results[0]
        confidence = best_result.similarity_score

        # Synthesize answer from top retrieved chunks
        synthesized_text = self._synthesize_answer(clean_query, results)

        sources = []
        for r in results:
            sources.append({
                "chunk_id": r.chunk_id,
                "section": r.header_path,
                "similarity_score": r.similarity_score,
                "snippet": r.content,
            })

        return {
            "query": query,
            "answer": synthesized_text,
            "top_section": best_result.header_path,
            "confidence": confidence,
            "sources": sources,
        }

    def _synthesize_answer(self, query: str, results: List[SearchResult]) -> str:
        """
        Combines and formats retrieved chunk content into a coherent response with citations.
        """
        primary = results[0]
        sec_name = primary.header_path.split(" > ")[-1] if " > " in primary.header_path else primary.header_path
        
        answer_parts = [
            f"Based on **{sec_name}**:\n",
            primary.content,
        ]

        # If secondary chunks provide complementary info from different or same section, incorporate key points
        if len(results) > 1 and results[1].similarity_score > 0.35:
            secondary = results[1]
            if secondary.content[:50] not in primary.content:
                sec_title = secondary.header_path.split(" > ")[-1] if " > " in secondary.header_path else secondary.header_path
                answer_parts.append(f"\n\n**Additional context from {sec_title}:**\n{secondary.content}")

        return "\n".join(answer_parts)
