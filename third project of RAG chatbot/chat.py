import argparse
import os
import sys
import time
from dotenv import load_dotenv

# Load .env file
load_dotenv()

import chromadb
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from evaluator import (
    calculate_summary,
    display_results,
    evaluate_question,
    extract_token_cost,
    resolve_api_key,
    resolve_eval_model,
    run_benchmark_evaluation,
)

# 1. Connect to the local vector database
DB_DIR = "./chroma_db"
COLLECTION_NAME = "alderbridge_employee_rulebook"

if not os.path.exists(DB_DIR):
    print(f"Error: Database directory '{DB_DIR}' not found. Please run rag_pdf.py first to build the index.")
    sys.exit(1)

print("Connecting to local Chroma database...")
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

# 2. Build MMR Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", 0) + 1
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        formatted.append(f"[Source {i} | Page {page} | Chunk #{chunk_id}]:\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


# 3. Setup Prompt Template
prompt_template = ChatPromptTemplate.from_template("""You are an expert HR assistant answering questions based solely on the official Alderbridge Employee Rulebook.

Official Context:
{context}

Employee Question:
{question}

Answer strictly based on the context above. If the information is not covered in the rulebook, clearly state that. Cite the applicable Policy ID or Section:""")

# 4. Initialize Gemini LLM with fallback
api_key = resolve_api_key()
preferred_model = os.getenv("GEMINI_MODEL")
fallback_models = [
    m for m in [preferred_model, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
    if m
]
# Preserve order while deduplicating
fallback_models = list(dict.fromkeys(fallback_models))

llm = None
active_model_name = None

if api_key:
    from langchain_google_genai import ChatGoogleGenerativeAI
    for m in fallback_models:
        try:
            test_llm = ChatGoogleGenerativeAI(model=m, temperature=0.1, google_api_key=api_key)
            llm = test_llm
            active_model_name = m
            print(f"Connected to Google Gemini ({m})!")
            break
        except Exception:
            continue

parser = StrOutputParser()


def run_rag_turn(question: str):
    """Retrieve context and generate an answer, returning `(docs, answer, cost)`."""
    if not llm:
        docs = retriever.invoke(question)
        return docs, "", 0.0

    docs = retriever.invoke(question)
    context_str = format_docs(docs)
    ai_msg = llm.invoke(prompt_template.invoke({"context": context_str, "question": question}))
    answer = parser.invoke(ai_msg)
    cost = extract_token_cost(ai_msg, active_model_name)
    return docs, answer, cost


def ask_chatbot(question: str, evaluate: bool = True, eval_history: list | None = None):
    """Answers an employee question using the local retriever, optional LLM, and RAG evaluation."""
    start = time.perf_counter()
    docs, answer, cost = run_rag_turn(question)
    delay = round(time.perf_counter() - start, 3)
    context_str = format_docs(docs)

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    if llm:
        print(f"\n[AI Answer]:\n{answer}\n")

        if evaluate:
            try:
                eval_model = resolve_eval_model(os.getenv("GEMINI_EVAL_MODEL") or active_model_name)
                eval_result = evaluate_question(
                    question=question,
                    retrieved_docs=docs,
                    answer=answer,
                    relevant_doc_ids=None,
                    reference_answer=None,
                    k=3,
                    cost=cost,
                    delay=delay,
                    api_key=api_key,
                    model=eval_model,
                )
                if eval_history is not None:
                    eval_history.append(eval_result)

                print("-" * 70)
                print("EVALUATION METRICS:")
                print(f"  • Groundedness : {eval_result['groundedness']}")
                print(f"  • Refusal      : {eval_result['refusal']}")
                print(f"  • Delay (s)    : {eval_result['delay_seconds']}")
                print(f"  • Est. Cost ($): {eval_result['cost']:.6f}")
                print(f"  • Judge Reason : {eval_result['judge_reason']}")
            except Exception as exc:
                print("-" * 70)
                print(f"EVALUATION METRICS: skipped due to judge error ({exc})")
    else:
        print("\n[Retrieved Policy Content (Exact Sources)]:")
        print(context_str)

    print("-" * 70)
    print("CITATIONS & SOURCES:")
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", 0) + 1
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        first_line = doc.page_content.strip().split("\n")[0]
        print(f"  • Source {i}: Page {page} (Chunk #{chunk_id}) -> {first_line[:80]}")
    print("=" * 70 + "\n")


def run_cli_benchmark():
    """Run the full benchmark evaluation suite in the terminal."""
    if not llm:
        print("Error: No Gemini API key configured. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env.")
        return
    eval_model = resolve_eval_model(os.getenv("GEMINI_EVAL_MODEL") or active_model_name)
    print(f"Running benchmark evaluation using judge model '{eval_model}'...")
    results, _ = run_benchmark_evaluation(
        run_rag_turn,
        k=3,
        api_key=api_key,
        model=eval_model,
    )
    display_results(results)


def interactive_chat(evaluate: bool = True):
    print("\n" + "=" * 70)
    print("  Alderbridge Employee Rulebook Chatbot  ")
    print("=" * 70)
    print("Ask any question about company policies, working hours, leave, expenses, etc.")
    print("Commands: 'eval' (run benchmark), 'summary' (session metrics), 'exit'/'quit'/'q'\n")

    eval_history = []

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            cmd = user_input.lower()
            if cmd in ["exit", "quit", "q"]:
                if eval_history:
                    print("\nSession Evaluation Summary:")
                    for metric, value in calculate_summary(eval_history).items():
                        print(f"  {metric}: {value}")
                print("Goodbye!")
                break
            if cmd == "eval":
                run_cli_benchmark()
                continue
            if cmd == "summary":
                if not eval_history:
                    print("No evaluated questions in this session yet.\n")
                else:
                    print("\nSession Evaluation Summary:")
                    for metric, value in calculate_summary(eval_history).items():
                        print(f"  {metric}: {value}")
                    print()
                continue
            ask_chatbot(user_input, evaluate=evaluate, eval_history=eval_history)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Alderbridge Employee Rulebook CLI Chatbot")
    arg_parser.add_argument("question", nargs="*", help="Optional single question to ask")
    arg_parser.add_argument("--eval", action="store_true", help="Run the full RAG benchmark evaluation suite")
    arg_parser.add_argument("--no-eval", action="store_true", help="Disable per-question LLM judge evaluation")
    args = arg_parser.parse_args()

    if args.eval:
        run_cli_benchmark()
    elif args.question:
        ask_chatbot(" ".join(args.question), evaluate=not args.no_eval)
    else:
        interactive_chat(evaluate=not args.no_eval)
