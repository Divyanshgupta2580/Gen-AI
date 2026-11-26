import os
import sys
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
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
fallback_models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
llm = None

if api_key:
    from langchain_google_genai import ChatGoogleGenerativeAI
    for m in fallback_models:
        try:
            test_llm = ChatGoogleGenerativeAI(model=m, temperature=0.1, google_api_key=api_key)
            llm = test_llm
            print(f"Connected to Google Gemini ({m})!")
            break
        except Exception:
            continue


def ask_chatbot(question: str):
    """Answers an employee question using the local retriever and optional LLM."""
    docs = retriever.invoke(question)
    context_str = format_docs(docs)

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    if llm:
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": question})
        print(f"\n[AI Answer]:\n{answer}\n")
    else:
        print("\n[Retrieved Policy Content (Exact Sources)]:")
        print(context_str)

    print("-" * 70)
    print("CITATIONS & SOURCES:")
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", 0) + 1
        first_line = doc.page_content.strip().split("\n")[0]
        print(f"  • Source {i}: Page {page} -> {first_line[:80]}")
    print("=" * 70 + "\n")


def interactive_chat():
    print("\n" + "=" * 70)
    print("  Alderbridge Employee Rulebook Chatbot  ")
    print("=" * 70)
    print("Ask any question about company policies, working hours, leave, expenses, etc.")
    print("Type 'exit' or 'quit' or 'q' to end the session.\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            ask_chatbot(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        ask_chatbot(query)
    else:
        interactive_chat()
