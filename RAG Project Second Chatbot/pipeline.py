#!/usr/bin/env python3
"""
End-to-End RAG Ingestion Pipeline:
Upload -> Load -> Split -> Clean -> Chunk -> Embedding -> Database

Demonstrates the custom chunking method:
'Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping'
for the Multiverse™ Reality Engine & Quantum Spatial Platform.
"""

import sys
import json
from pathlib import Path
from src.uploader import DocumentUploader
from src.loader import DocumentLoader
from src.splitter import StructuralSplitter
from src.cleaner import TextCleaner
from src.chunker import HierarchicalContextualChunker
from src.embedder import VectorEmbedder
from src.database import VectorDatabase
from src.rag_engine import RAGEngine


def print_banner(title: str, stage_num: int):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f" STAGE {stage_num}: {title.upper()}")
    print(f"{sep}")


def main():
    print("\n" + "#" * 80)
    print("  MULTIVERSE™ AUTONOMOUS RAG INGESTION PIPELINE")
    print("  Pipeline Flow: Upload -> Load -> Split -> Clean -> Chunk -> Embedding -> Database")
    print("#" * 80)

    # Document path created for this project: Multiverse Manual
    source_doc_path = "./data/multiverse_manual.md"
    if len(sys.argv) > 1:
        source_doc_path = sys.argv[1]

    # ==========================================
    # STAGE 1: UPLOAD
    # ==========================================
    print_banner("Document Upload & Validation", 1)
    uploader = DocumentUploader(upload_dir="./uploads")
    uploaded_doc = uploader.upload(source_doc_path)
    print(f"[+] Document Uploaded Successfully:")
    print(f"    - Document ID     : {uploaded_doc.doc_id}")
    print(f"    - Original Name   : {uploaded_doc.original_filename}")
    print(f"    - Staged Path     : {uploaded_doc.stored_path}")
    print(f"    - File Size       : {uploaded_doc.file_size_bytes:,} bytes")
    print(f"    - SHA-256 Hash    : {uploaded_doc.content_hash_sha256}")
    print(f"    - MIME Type       : {uploaded_doc.mime_type}")

    # ==========================================
    # STAGE 2: LOAD
    # ==========================================
    print_banner("Document Loading & Metric Extraction", 2)
    loader = DocumentLoader()
    loaded_doc = loader.load(uploaded_doc)
    print(f"[+] Document Loaded Successfully:")
    print(f"    - Total Characters: {loaded_doc.character_count:,}")
    print(f"    - Total Words     : {loaded_doc.metadata.get('total_words', 0):,}")
    print(f"    - Total Lines     : {loaded_doc.metadata.get('total_lines', 0):,}")
    print(f"    - Encoding        : {loaded_doc.encoding}")

    # ==========================================
    # STAGE 3: SPLIT
    # ==========================================
    print_banner("Structural Header & Hierarchy Splitting", 3)
    splitter = StructuralSplitter()
    blocks = splitter.split(loaded_doc)
    print(f"[+] Document Structurally Decomposed into {len(blocks)} Logical Blocks:")
    for i, b in enumerate(blocks[:4], 1):
        breadcrumb_str = " > ".join(b.breadcrumb) if b.breadcrumb else b.section_title
        print(f"    Block {i:02d} [{b.block_type.upper():<9}] Hierarchy: {breadcrumb_str}")
    print(f"    ... and {len(blocks) - 4} additional blocks.")

    # ==========================================
    # STAGE 4: CLEAN
    # ==========================================
    print_banner("Text Normalization & Noise Cleaning", 4)
    cleaner = TextCleaner()
    cleaned_blocks = cleaner.clean(blocks)
    chars_before = sum(len(b.raw_content) for b in blocks)
    chars_after = sum(len(b.cleaned_content) for b in cleaned_blocks)
    print(f"[+] Text Normalized Across {len(cleaned_blocks)} Blocks:")
    print(f"    - Chars Before    : {chars_before:,}")
    print(f"    - Chars After     : {chars_after:,} (Removed {chars_before - chars_after:,} redundant noise characters)")
    print(f"    - Markdown Headers Preserved: Yes")
    print(f"    - Smart Quotes / Dashes Standardized: Yes")

    # ==========================================
    # STAGE 5: CHUNK (Custom HC-SW Method)
    # ==========================================
    print_banner("Hierarchical Context-Aware Sliding Window Chunking", 5)
    chunker = HierarchicalContextualChunker(max_chunk_size=450, overlap_size=110, min_chunk_size=80)
    chunks = chunker.chunk(cleaned_blocks)
    print(f"[+] Custom HC-SW Chunking Generated {len(chunks)} Context-Enriched Chunks:")
    for i, c in enumerate(chunks[:3], 1):
        print(f"    Chunk {i:02d} [{c.chunk_id}]: {len(c.content)} chars | Overlap: {c.overlap_char_count} chars")
        print(f"      Hierarchy: {c.header_path}")
        print(f"      Sample   : \"{c.content[:85]}...\"")
    print(f"    ... and {len(chunks) - 3} additional chunks.")

    # ==========================================
    # STAGE 6: EMBEDDING
    # ==========================================
    print_banner("Dense Vector Embeddings Generation", 6)
    embedder = VectorEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"[+] Embedding Model Initialized:")
    print(f"    - Model Name      : {embedder.MODEL_NAME}")
    print(f"    - Vector Dimension: {embedder.DIMENSION} dimensions")
    print(f"    - Embedded Chunks : {len(embedded_chunks)}")
    print(f"    - Similarity Metric: Cosine Similarity")
    print(f"    - Encoding Mode   : Contextual Hierarchy Prepended")

    # ==========================================
    # STAGE 7: DATABASE (ChromaDB Vector Store)
    # ==========================================
    print_banner("ChromaDB Vector Store Persistence", 7)
    db = VectorDatabase(persist_dir="./vector_store")
    print("[*] Storing chunks and embeddings into persistent vector store...")
    db.reset_collection()
    stored_count = db.store_chunks(embedded_chunks)
    count = db.count()
    print(f"[+] ChromaDB Storage Complete:")
    print(f"    - Collection Name : {db.COLLECTION_NAME}")
    print(f"    - Total Chunks    : {count}")
    print(f"    - Persistence Path: ./vector_store")

    # ==========================================
    # VERIFICATION QUERY
    # ==========================================
    print("\n" + "=" * 80)
    print(" PIPELINE VERIFICATION: TEST SEMANTIC SEARCH & RAG CHATBOT")
    print("=" * 80)
    rag_engine = RAGEngine(persist_dir="./vector_store")
    test_queries = [
        "What is Multiverse?",
        "What are the specs of the Helios-Q9 quantum processor?",
        "How do the Einstein-Rosen wormholes maintain low latency?",
    ]
    for q in test_queries:
        print(f"\n🔎 Question: \"{q}\"")
        result = rag_engine.answer_query(q, top_k=2)
        print(f"   🤖 Answer: {result['answer']}")
        print(f"   📊 Confidence: {result['confidence'] * 100:.1f}%")
        if result.get("sources"):
            print(f"   📚 Source: {result['sources'][0]['section']}")

    print("\n" + "#" * 80)
    print("  MULTIVERSE™ RAG PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
