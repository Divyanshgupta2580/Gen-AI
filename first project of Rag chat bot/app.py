"""
PDF Tutor AI - Lightweight Hybrid RAG Chatbot
A local Streamlit educational chatbot powered by TF-IDF lexical vectorization,
rank-bm25 keyword retrieval, ChromaDB persistent storage, Reciprocal Rank Fusion (RRF),
and optional Google Gemini API natural language tutoring.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from config.settings import (
    APP_TITLE,
    APP_SUBTITLE,
    DEFAULT_GEMINI_MODEL,
    AVAILABLE_GEMINI_MODELS,
    GEMINI_API_KEY_ENV,
)
from tutor.tutor_engine import TutorEngine
from tutor.learning_modes import LEARNING_MODES, LEARNING_MODE_NAMES
from utils.validators import ValidationError
from ingestion.pdf_loader import PDFProcessingError

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "tutor_engine" not in st.session_state:
        initial_api_key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
        st.session_state.tutor_engine = TutorEngine(
            gemini_api_key=initial_api_key,
            gemini_model=DEFAULT_GEMINI_MODEL,
        )
    if "document_info" not in st.session_state:
        st.session_state.document_info = None
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = "Explain Concept"
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


init_session_state()
engine: TutorEngine = st.session_state.tutor_engine

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🎓 PDF Tutor AI")
    st.caption("Lightweight Hybrid RAG Local Chatbot")
    st.markdown("---")
    
    # 1. PDF File Ingestion
    st.subheader("📄 Document Ingestion")
    pdf_path_input = st.text_input(
        "Absolute PDF File Path:",
        placeholder="/Users/username/Documents/sample.pdf",
        help="Enter the absolute or full filesystem path to your educational PDF.",
    )
    
    col_load, col_clear = st.columns([1, 1])
    with col_load:
        load_clicked = st.button("📥 Load PDF", use_container_width=True, type="primary")
    with col_clear:
        clear_chat_clicked = st.button("🗑️ Clear Chat", use_container_width=True)
        
    if clear_chat_clicked:
        engine.conversation.clear()
        st.session_state.chat_messages = []
        st.success("Chat history cleared.")
        st.rerun()

    if load_clicked:
        if not pdf_path_input.strip():
            st.error("Please enter a valid PDF path.")
        else:
            with st.spinner("Extracting text, chunking, and building hybrid indexes..."):
                try:
                    doc_stats = engine.load_document(pdf_path_input.strip())
                    st.session_state.document_info = doc_stats
                    st.session_state.chat_messages = []
                    st.success(f"Loaded '{doc_stats['document_name']}' successfully!")
                    st.rerun()
                except (ValidationError, PDFProcessingError) as e:
                    st.error(f"Failed to load PDF: {e}")
                except Exception as e:
                    st.error(f"Unexpected error loading document: {e}")

    # Active Document Information
    if st.session_state.document_info:
        info = st.session_state.document_info
        st.markdown("---")
        st.subheader("📊 Active Document")
        st.markdown(f"**File:** `{info['document_name']}`")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Pages", info["total_pages"])
        with col_m2:
            st.metric("Chunks", info["total_chunks"])
        st.caption(f"Fingerprint: `{info['doc_hash'][:12]}...`")
    else:
        st.info("No document loaded yet. Provide a PDF path above.")

    st.markdown("---")
    
    # 2. Tutoring Learning Mode
    st.subheader("🎯 Tutoring Mode")
    selected_mode = st.selectbox(
        "Choose learning objective:",
        options=LEARNING_MODE_NAMES,
        index=LEARNING_MODE_NAMES.index(st.session_state.selected_mode),
        help="Select pedagogical behavior for questions.",
    )
    st.session_state.selected_mode = selected_mode
    mode_obj = LEARNING_MODES[selected_mode]
    st.caption(f"{mode_obj.icon} **{mode_obj.name}:** {mode_obj.description}")

    st.markdown("---")
    
    # 3. System Engine & API Status
    st.subheader("⚙️ System Status")
    
    # Retrieval Engine Status
    retrieval_status = "🟢 Hybrid (TF-IDF + BM25 + ChromaDB + RRF)" if engine.is_document_loaded else "⚪ Idle"
    st.markdown(f"**Retrieval Engine:** {retrieval_status}")
    
    # API & Answer Generation Status
    api_key_env_present = bool(os.getenv(GEMINI_API_KEY_ENV, "").strip())
    sidebar_api_key = st.text_input(
        "Gemini API Key (optional):",
        type="password",
        value=engine.gemini_client.api_key,
        placeholder="AIzaSy...",
        help="Optional. If omitted, the chatbot operates in 100% free Retrieval-Only mode.",
    )
    if sidebar_api_key != engine.gemini_client.api_key:
        engine.set_gemini_api_key(sidebar_api_key)

    selected_model = st.selectbox(
        "Gemini Model:",
        options=AVAILABLE_GEMINI_MODELS,
        index=0,
    )
    engine.set_gemini_model(selected_model)

    if engine.is_gemini_available:
        st.success("🟢 Answer Generation: Gemini API Connected")
    else:
        st.warning("🟡 Mode: Retrieval-Only (No API Key)")
        st.caption("Displays retrieved source passages with page citations and relevance scores.")


# ----------------- MAIN INTERFACE -----------------
st.title(f"{APP_TITLE}")
st.markdown(f"*{APP_SUBTITLE}*")

# Top Banner: Document Status
if st.session_state.document_info:
    info = st.session_state.document_info
    mode_display = LEARNING_MODES[st.session_state.selected_mode]
    st.info(
        f"📖 **Active Document:** `{info['document_name']}` ({info['total_pages']} pages, {info['total_chunks']} chunks)  |  "
        f"**Active Mode:** {mode_display.icon} `{mode_display.name}`  |  "
        f"**Status:** {'🤖 Natural Language Tutor (Gemini)' if engine.is_gemini_available else '🔍 Retrieval-Only'}"
    )
else:
    st.warning("👈 Please enter a valid PDF path in the sidebar and click **'Load PDF'** to begin tutoring.")
    with st.expander("ℹ️ How This Lightweight Hybrid RAG Chatbot Works", expanded=True):
        st.markdown(
            """
            - **100% Lightweight & Local:** Runs on standard CPU. Zero local neural networks, no Ollama, no heavy PyTorch/transformers.
            - **Page-Aware Ingestion:** Uses PyMuPDF to extract text page-by-page and tracks true citations.
            - **TF-IDF Vector Retrieval:** Computes lexical document vectors and cosine similarity using scikit-learn.
            - **ChromaDB Storage:** Persists vector embeddings and metadata locally in SQLite without downloading models.
            - **BM25 Keyword Retrieval:** Matches technical terminology and acronyms with BM25Okapi.
            - **Reciprocal Rank Fusion (RRF):** Fuses dense vector and sparse keyword rankings using the formula $RRF(d) = \\sum \\frac{1}{k + \\text{rank}(d)}$ with $k=60$.
            - **Dual Answer Modes:** Connect your `GEMINI_API_KEY` for natural conversational answers, or use **Retrieval-Only Mode** with zero API keys!
            """
        )

# Render Chat History
for msg in engine.conversation.messages:
    if msg.role == "user":
        with st.chat_message("user"):
            st.markdown(f"**[{msg.mode}]** {msg.content}")
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)
            
            # Display Page Citations
            if msg.citations:
                badges = " ".join([f"`Page {p}`" for p in msg.citations])
                st.markdown(f"📌 **Citations:** {badges}")
                
            # Expandable Retrieved Passages
            if msg.retrieved_chunks:
                with st.expander(f"🔍 Inspect Retrieved Passages ({len(msg.retrieved_chunks)} chunks)"):
                    for i, r in enumerate(msg.retrieved_chunks, 1):
                        tfidf_rank = r.get("sources", {}).get("tfidf", {}).get("rank", "N/A")
                        tfidf_score = r.get("sources", {}).get("tfidf", {}).get("score", 0.0)
                        bm25_rank = r.get("sources", {}).get("bm25", {}).get("rank", "N/A")
                        bm25_score = r.get("sources", {}).get("bm25", {}).get("score", 0.0)
                        
                        st.markdown(
                            f"**Passage {i} — Page {r['page_number']}**  \n"
                            f"*RRF Score:* `{r['rrf_score']:.5f}` | "
                            f"*TF-IDF Rank:* `#{tfidf_rank} ({tfidf_score:.4f})` | "
                            f"*BM25 Rank:* `#{bm25_rank} ({bm25_score:.4f})`"
                        )
                        st.text(r["text"])
                        st.divider()

# Question Input
if user_question := st.chat_input("Ask a question about the PDF..."):
    if not engine.is_document_loaded:
        st.error("Please load a PDF document from the sidebar first!")
    else:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**[{st.session_state.selected_mode}]** {user_question}")
            
        # Process and generate response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and synthesizing tutor response..."):
                result = engine.answer_query(
                    user_query=user_question,
                    mode=st.session_state.selected_mode,
                )
                st.markdown(result["answer"])
                
                # Citations
                if result["citations"]:
                    badges = " ".join([f"`Page {p}`" for p in result["citations"]])
                    st.markdown(f"📌 **Citations:** {badges}")
                    
                # Expandable inspection
                if result["retrieved_chunks"]:
                    with st.expander(f"🔍 Inspect Retrieved Passages ({len(result['retrieved_chunks'])} chunks)"):
                        for i, r in enumerate(result["retrieved_chunks"], 1):
                            tfidf_rank = r.get("sources", {}).get("tfidf", {}).get("rank", "N/A")
                            tfidf_score = r.get("sources", {}).get("tfidf", {}).get("score", 0.0)
                            bm25_rank = r.get("sources", {}).get("bm25", {}).get("rank", "N/A")
                            bm25_score = r.get("sources", {}).get("bm25", {}).get("score", 0.0)
                            
                            st.markdown(
                                f"**Passage {i} — Page {r['page_number']}**  \n"
                                f"*RRF Score:* `{r['rrf_score']:.5f}` | "
                                f"*TF-IDF Rank:* `#{tfidf_rank} ({tfidf_score:.4f})` | "
                                f"*BM25 Rank:* `#{bm25_rank} ({bm25_score:.4f})`"
                            )
                            st.text(r["text"])
                            st.divider()
        st.rerun()
