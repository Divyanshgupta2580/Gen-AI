"""
Reciprocal Rank Fusion (RRF) Module.

Combines rankings from multiple heterogeneous retrieval systems
(lexical vector TF-IDF + keyword BM25) into a single unified ranking.

Formula:
    RRF(d) = SUM( 1 / (k + rank_i(d)) )
where:
    - k is the rank constant (standard default: 60)
    - rank_i(d) is the 1-based position of document d in ranking i
"""

from typing import List, Dict, Any, Tuple, Optional
from config.settings import RRF_K
from ingestion.chunker import DocumentChunk


def reciprocal_rank_fusion(
    ranked_lists: List[Tuple[str, List[Tuple[DocumentChunk, float]]]],
    k: int = RRF_K,
    top_k_final: int = 5,
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion over multiple ranked result sets.
    
    Args:
        ranked_lists: List of (source_name, [(chunk, raw_score), ...])
                      where items in the inner list are pre-sorted by rank descending.
        k: Smoothing constant to control the weight of high rankings (default: 60).
        top_k_final: Number of final deduplicated candidates to return.
        
    Returns:
        List of candidate dictionaries ordered by descending RRF score, containing:
        - "chunk": DocumentChunk
        - "rrf_score": float
        - "sources": dict mapping source_name -> {"rank": int, "score": float}
    """
    if not ranked_lists:
        return []
        
    chunk_map: Dict[str, DocumentChunk] = {}
    rrf_scores: Dict[str, float] = {}
    source_details: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for source_name, results in ranked_lists:
        for rank_zero_indexed, (chunk, score) in enumerate(results):
            cid = chunk.chunk_id
            rank_1_indexed = rank_zero_indexed + 1
            
            # Store chunk reference
            if cid not in chunk_map:
                chunk_map[cid] = chunk
                rrf_scores[cid] = 0.0
                source_details[cid] = {}
                
            # Add RRF reciprocal rank score
            rrf_contribution = 1.0 / (k + rank_1_indexed)
            rrf_scores[cid] += rrf_contribution
            
            source_details[cid][source_name] = {
                "rank": rank_1_indexed,
                "score": float(score),
                "rrf_contribution": rrf_contribution,
            }
            
    # Sort candidates by combined RRF score descending
    sorted_chunk_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )
    
    final_results: List[Dict[str, Any]] = []
    for cid in sorted_chunk_ids[:top_k_final]:
        final_results.append({
            "chunk": chunk_map[cid],
            "rrf_score": rrf_scores[cid],
            "sources": source_details[cid],
            "chunk_id": cid,
            "page_number": chunk_map[cid].page_number,
            "document_name": chunk_map[cid].document_name,
            "text": chunk_map[cid].text,
        })
        
    return final_results
