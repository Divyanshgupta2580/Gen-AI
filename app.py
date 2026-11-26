import os
import streamlit as st
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="Alderbridge Rulebook AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DB_DIR = "./chroma_db"
COLLECTION_NAME = "alderbridge_employee_rulebook"
DEFAULT_KEY = os.getenv("GOOGLE_API_KEY", "")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_components(api_key: str, model_name: str):
    """Initializes and caches the local vector store and Gemini LLM."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    client = chromadb.PersistentClient(path=DB_DIR)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template("""You are an expert HR Policy Advisor for Alderbridge Office Systems.
Answer the employee's question accurately, clearly, and solely based on the official Alderbridge Employee Rulebook context provided below.

Official Policy Context:
{context}

Employee Question:
{question}

Guidelines:
1. Provide a direct, helpful, and policy-compliant answer.
2. Cite the exact Policy Number (e.g., Policy 02.01) and Section name whenever available.
3. If the context does not contain the information, state that clearly instead of speculating.""")

    chain = prompt | llm | StrOutputParser()

    return retriever, chain


# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/policy-document.png", width=64)
    st.title("Rulebook Settings")
    
    api_key = st.text_input("Gemini API Key", value=DEFAULT_KEY, type="password")
    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"],
        index=0
    )

    st.markdown("---")
    st.subheader("System Status")
    st.success("🟢 Local Chroma DB: Active")
    st.info("📄 Document: Alderbridge-Employee-Rulebook.pdf")
    st.markdown("**Chunking:** Semantic (Sentence Distance)")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2` (Local)")
    st.markdown("**Distance Space:** Cosine Similarity")

    st.markdown("---")
    st.subheader("Quick Prompts")
    quick_queries = [
        "What are standard office hours and breaks?",
        "How many days of paid sick leave do I get?",
        "What are the rules on gifts and conflicts of interest?",
        "Can I work remotely, and what is the policy?",
        "What is the notice period for resignation?"
    ]

# Main UI
st.markdown("<div class='main-header'>📖 Alderbridge Employee Rulebook AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Instant, verified answers grounded in the official company policies.</div>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Alderbridge HR Policy Assistant. Ask me anything about working hours, paid leave, workplace safety, reporting misconduct, business expenses, or separation procedures."}
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Cited Rulebook Clauses"):
                for s in message["sources"]:
                    st.markdown(f"<div class='source-box'><span class='badge'>Page {s['page']}</span><b>{s['header']}</b><br>{s['text']}</div>", unsafe_allow_html=True)

# Function to process user question
def handle_query(user_query: str):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching official rulebook & generating answer..."):
            try:
                retriever, chain = load_rag_components(api_key, model_name)
                docs = retriever.invoke(user_query)

                # Format context
                context_str = "\n\n".join([f"[Page {d.metadata.get('page', 0)+1}]:\n{d.page_content.strip()}" for d in docs])
                answer = chain.invoke({"context": context_str, "question": user_query})

                # Prepare sources for expander
                sources = []
                for d in docs:
                    lines = d.page_content.strip().split("\n")
                    first_line = lines[0] if lines else "Clause"
                    body = "\n".join(lines[1:4]) if len(lines) > 1 else d.page_content.strip()[:180]
                    sources.append({
                        "page": d.metadata.get("page", 0) + 1,
                        "header": first_line,
                        "text": body
                    })

                st.markdown(answer)
                with st.expander("📚 View Cited Rulebook Clauses"):
                    for s in sources:
                        st.markdown(f"<div class='source-box'><span class='badge'>Page {s['page']}</span><b>{s['header']}</b><br>{s['text']}</div>", unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Error answering query: {e}")


# Quick buttons in sidebar
with st.sidebar:
    for q in quick_queries:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            handle_query(q)

# Chat input
if prompt := st.chat_input("Ask a question about the Employee Rulebook..."):
    handle_query(prompt)
