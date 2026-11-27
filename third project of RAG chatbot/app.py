import json
import os
import time
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

from evaluator import (
    ALDERBRIDGE_BENCHMARK,
    calculate_summary,
    evaluate_question,
    extract_token_cost,
    lookup_benchmark_case,
    run_benchmark_evaluation,
)

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
DEFAULT_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL_OPTIONS = list(dict.fromkeys([
    DEFAULT_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]))
EVAL_MODEL_OPTIONS = ["Same as Chat Model", "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]

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
    """Initializes and caches the local vector store, prompt template, and Gemini LLM."""
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

    llm = (
        ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1
        )
        if api_key
        else None
    )

    return retriever, prompt, llm


def run_rag_turn(retriever, prompt, llm, query: str, model_name: str):
    """Execute retrieval + generation and return `(docs, answer, cost, delay)`."""
    start = time.perf_counter()
    docs = retriever.invoke(query)
    context_str = "\n\n".join(
        [f"[Page {d.metadata.get('page', 0) + 1}]:\n{d.page_content.strip()}" for d in docs]
    )
    ai_message = llm.invoke(prompt.invoke({"context": context_str, "question": query}))
    answer = StrOutputParser().invoke(ai_message)
    delay = round(time.perf_counter() - start, 3)
    cost = extract_token_cost(ai_message, model_name)
    return docs, answer, cost, delay


def render_eval_expander(eval_result: dict):
    """Render per-response RAG evaluation metrics inside an expander."""
    with st.expander("📊 RAG Evaluation Metrics (0/1 & MRR)", expanded=True):
        c1, c2, c3 = st.columns(3)
        hit = eval_result.get("hit_rate_at_k")
        mrr = eval_result.get("mrr")
        groundedness = eval_result.get("groundedness")
        c1.metric("Hit Rate @ 3 (0/1)", str(hit) if hit is not None else "N/A")
        c2.metric("MRR", f"{mrr:.2f}" if mrr is not None else "N/A")
        c3.metric("Groundedness (0/1)", str(groundedness) if groundedness is not None else "N/A")

        c4, c5, c6, c7 = st.columns(4)
        correctness = eval_result.get("correctness")
        c4.metric("Correctness (0/1)", str(correctness) if correctness is not None else "N/A")
        c5.metric("Refusal (0/1)", str(eval_result.get("refusal", 0)))
        c6.metric("Delay", f"{eval_result.get('delay_seconds', 0.0):.2f}s")
        c7.metric("Est. Cost", f"${eval_result.get('cost', 0.0):.6f}")

        if eval_result.get("judge_reason"):
            st.caption(f"**Judge Reason:** {eval_result['judge_reason']}")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Alderbridge HR Policy Assistant. Ask me anything about working hours, paid leave, workplace safety, reporting misconduct, business expenses, or separation procedures."
        }
    ]
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/policy-document.png", width=64)
    st.title("Rulebook Settings")

    mode = st.radio("Workspace Mode", ["💬 Chat Assistant", "📊 Evaluation Suite"])

    api_key = st.text_input("Gemini API Key", value=DEFAULT_KEY, type="password")
    model_name = st.selectbox("Gemini Model", MODEL_OPTIONS, index=0)

    st.markdown("---")
    st.subheader("Evaluation Settings")
    enable_live_eval = st.toggle(
        "Live RAG Judge",
        value=True,
        help="Automatically evaluate each answer for Groundedness, Refusal, Latency, and Cost."
    )
    eval_model_choice = st.selectbox("Judge Model", EVAL_MODEL_OPTIONS, index=0)
    eval_model = model_name if eval_model_choice == "Same as Chat Model" else eval_model_choice
    session_metrics_slot = st.empty()


def render_sidebar_session_metrics():
    """Render live session evaluation summary in the sidebar."""
    if not st.session_state.eval_results:
        return
    summary = calculate_summary(st.session_state.eval_results)
    h_avg = summary["Hit Rate at K"]
    m_avg = summary["MRR"]
    g_avg = summary["Groundedness"]
    with session_metrics_slot.container():
        st.markdown("**Session Metrics**")
        st.caption(
            f"Turns: {summary['Total Questions']} | "
            f"Hit Rate: {f'{h_avg:.2f}' if h_avg is not None else 'N/A'} | "
            f"MRR: {f'{m_avg:.2f}' if m_avg is not None else 'N/A'} | "
            f"Groundedness: {f'{g_avg:.2f}' if g_avg is not None else 'N/A'} | "
            f"Refusal: {summary['Refusal Rate']:.0%} | "
            f"Delay: {summary['Average Delay (seconds)']:.2f}s"
        )


render_sidebar_session_metrics()

with st.sidebar:
    st.markdown("---")
    st.subheader("System Status")
    if os.path.exists(DB_DIR):
        st.success("🟢 Local Chroma DB: Active")
    else:
        st.warning("🟡 Local Chroma DB missing — run `python rag_pdf.py` first.")
    st.info("📄 Document: Alderbridge-Employee-Rulebook.pdf")
    st.markdown("**Chunking:** Semantic (Sentence Distance)")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2` (Local)")
    st.markdown("**Distance Space:** Cosine Similarity")

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


# Function to process user question
def handle_query(user_query: str):
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to ask questions.")
        return
    if not os.path.exists(DB_DIR):
        st.error(f"Local Chroma DB not found at '{DB_DIR}'. Run `python rag_pdf.py` first to build the index.")
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching official rulebook & generating answer..."):
            try:
                retriever, prompt, llm = load_rag_components(api_key, model_name)
                docs, answer, cost, delay = run_rag_turn(retriever, prompt, llm, user_query, model_name)

                # Prepare sources for expander
                sources = []
                for d in docs:
                    lines = d.page_content.strip().split("\n")
                    first_line = lines[0] if lines else "Clause"
                    body = "\n".join(lines[1:4]) if len(lines) > 1 else d.page_content.strip()[:180]
                    sources.append({
                        "page": d.metadata.get("page", 0) + 1,
                        "chunk_id": d.metadata.get("chunk_id", "N/A"),
                        "header": first_line,
                        "text": body
                    })

                st.markdown(answer)
                with st.expander("📚 View Cited Rulebook Clauses"):
                    for s in sources:
                        st.markdown(
                            f"<div class='source-box'><span class='badge'>Page {s['page']} · Chunk #{s['chunk_id']}</span>"
                            f"<b>{s['header']}</b><br>{s['text']}</div>",
                            unsafe_allow_html=True
                        )

                eval_result = None
                try:
                    matched_case = lookup_benchmark_case(user_query)
                    eval_result = evaluate_question(
                        question=user_query,
                        retrieved_docs=docs,
                        answer=answer,
                        relevant_doc_ids=matched_case.get("relevant_doc_ids") if matched_case else None,
                        reference_answer=matched_case.get("reference_answer") if matched_case else None,
                        k=3,
                        cost=cost,
                        delay=delay,
                        api_key=api_key,
                        model=eval_model,
                        run_judge=enable_live_eval,
                        infer_relevant_from_judge=(
                            matched_case is None or matched_case.get("relevant_doc_ids") is None
                        ),
                    )
                    st.session_state.eval_results.append(eval_result)
                    render_sidebar_session_metrics()
                    render_eval_expander(eval_result)
                except Exception as eval_exc:
                    st.caption(f"⚠️ Evaluation skipped: {eval_exc}")

                message_entry = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
                if eval_result is not None:
                    message_entry["evaluation"] = eval_result
                st.session_state.messages.append(message_entry)
            except Exception as e:
                st.error(f"Error answering query: {e}")


if mode == "💬 Chat Assistant":
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Cited Rulebook Clauses"):
                    for s in message["sources"]:
                        chunk_badge = f" · Chunk #{s['chunk_id']}" if "chunk_id" in s else ""
                        st.markdown(
                            f"<div class='source-box'><span class='badge'>Page {s['page']}{chunk_badge}</span>"
                            f"<b>{s['header']}</b><br>{s['text']}</div>",
                            unsafe_allow_html=True
                        )
            if "evaluation" in message:
                render_eval_expander(message["evaluation"])

    # Quick buttons in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("Quick Prompts")
        for q in quick_queries:
            if st.button(q, key=f"btn_{q}", width="stretch"):
                handle_query(q)

    # Chat input
    if prompt_input := st.chat_input("Ask a question about the Employee Rulebook..."):
        handle_query(prompt_input)

else:
    st.subheader("🧪 Alderbridge RAG Benchmark Evaluation")
    st.write(
        "Run automated retrieval (`Hit Rate @ K`, `MRR`), refusal detection, latency, cost, "
        "and Gemini LLM-judge (`Groundedness`, `Correctness`) evaluation across the benchmark dataset."
    )

    with st.expander(f"Preview Benchmark Dataset ({len(ALDERBRIDGE_BENCHMARK)} questions)"):
        st.dataframe(
            [
                {
                    "Question": c["question"],
                    "Relevant Chunk IDs": str(c["relevant_doc_ids"]) if c["relevant_doc_ids"] is not None else "Out-of-scope (Refusal)",
                    "Reference Answer": c["reference_answer"],
                }
                for c in ALDERBRIDGE_BENCHMARK
            ],
            width="stretch",
            hide_index=True,
        )

    if st.button("🚀 Run Full Benchmark Evaluation", type="primary"):
        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar first.")
        elif not os.path.exists(DB_DIR):
            st.error(f"Local Chroma DB not found at '{DB_DIR}'. Run `python rag_pdf.py` first to build the index.")
        else:
            try:
                retriever, prompt, llm = load_rag_components(api_key, model_name)
                progress_bar = st.progress(0.0, text="Starting evaluation...")

                def _on_progress(idx: int, total: int, q: str):
                    progress_bar.progress(idx / total, text=f"Evaluating {idx}/{total}: {q}")

                def _ask(q: str):
                    docs, answer, cost, _ = run_rag_turn(retriever, prompt, llm, q, model_name)
                    return docs, answer, cost

                results, summary = run_benchmark_evaluation(
                    _ask,
                    k=3,
                    api_key=api_key,
                    model=eval_model,
                    run_judge=True,
                    progress_callback=_on_progress,
                )
                progress_bar.empty()
                st.session_state.benchmark_results = results
                st.session_state.benchmark_summary = summary
            except Exception as e:
                st.error(f"Benchmark evaluation failed: {e}")

    if "benchmark_summary" in st.session_state and "benchmark_results" in st.session_state:
        summary = st.session_state.benchmark_summary
        results = st.session_state.benchmark_results

        st.markdown("### Final Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hit Rate @ 3", f"{summary['Hit Rate at K']:.2f}" if summary["Hit Rate at K"] is not None else "N/A")
        m2.metric("MRR", f"{summary['MRR']:.2f}" if summary["MRR"] is not None else "N/A")
        m3.metric("Groundedness", f"{summary['Groundedness']:.2f}" if summary["Groundedness"] is not None else "N/A")
        m4.metric("Correctness", f"{summary['Correctness']:.2f}" if summary["Correctness"] is not None else "N/A")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Refusal Rate", f"{summary['Refusal Rate']:.0%}" if summary["Refusal Rate"] is not None else "N/A")
        m6.metric("Avg Cost ($)", f"${summary['Average Cost']:.6f}" if summary["Average Cost"] is not None else "N/A")
        m7.metric("Avg Delay (s)", f"{summary['Average Delay (seconds)']:.2f}s" if summary["Average Delay (seconds)"] is not None else "N/A")
        m8.metric("Total Questions", str(summary["Total Questions"]))

        st.markdown("### Per-Question Results")
        st.dataframe(results, width="stretch", hide_index=True)

        report_payload = json.dumps({"summary": summary, "results": results}, indent=2)
        st.download_button(
            "⬇️ Download Evaluation Report (JSON)",
            data=report_payload,
            file_name="rag_evaluation_report.json",
            mime="application/json",
        )
