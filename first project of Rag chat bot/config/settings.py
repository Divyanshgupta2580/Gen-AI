# Configuration settings for PDF Tutor AI - Lightweight Hybrid RAG Chatbot
import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data and persistence directories
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ingestion & Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval parameters
RRF_K = 60
TOP_K_TFIDF = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5  # Returns 4-5 most relevant chunks

# LLM / Gemini parameters
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
AVAILABLE_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# Application metadata
APP_TITLE = "PDF Tutor AI"
APP_SUBTITLE = "Lightweight Hybrid RAG Educational Chatbot"
APP_VERSION = "1.0.0"
