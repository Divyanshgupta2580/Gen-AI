import os
import re
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def clean_pdf_text(text: str) -> str:
    """
    Cleans raw PDF extracted text:
    1. Removes repetitive headers and footers/page numbers.
    2. Reconnects hyphenated line-broken words.
    3. Formats Section and Policy numbers into structured markdown headings.
    4. Merges soft line wraps within sentences.
    5. Normalizes whitespace and redundant newlines.
    """
    # 1. Remove running headers
    text = re.sub(
        r'(?i)ALDERBRIDGE\s+OFFICE\s+SYSTEMS\s*\n\s*(?:EMPLOYEE\s+RULEBOOK\s*/\s*VERSION\s*[\d\.]+\s*\n)?',
        '',
        text
    )

    # 2. Remove running footers and page numbering
    text = re.sub(
        r'(?i)FICTIONAL\s+COMPANY\s*/\s*EMPLOYEE\s+RULEBOOK\s*\n\s*Page\s*\d+\s*of\s*\d+',
        '',
        text
    )
    text = re.sub(r'(?i)^\s*Page\s*\d+\s*of\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 3. Heal hyphenated line breaks across lines
    text = re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', text)

    # 4. Standardize Section and Policy numbers into structured markdown headings
    text = re.sub(r'^\s*SECTION\s+(\d+)\s*\n\s*([^\n]+)', r'\n### SECTION \1: \2\n', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(\d{2}\.\d{2})\s*\n\s*([^\n]+)', r'\n#### Policy \1: \2\n', text, flags=re.MULTILINE)

    # 5. Join soft line breaks in body text
    lines = text.split('\n')
    cleaned_lines = []
    current_para = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                cleaned_lines.append(' '.join(current_para))
                current_para = []
            cleaned_lines.append('')
        elif stripped.startswith('#') or stripped.startswith('*') or stripped.startswith('-'):
            if current_para:
                cleaned_lines.append(' '.join(current_para))
                current_para = []
            cleaned_lines.append(stripped)
        else:
            current_para.append(stripped)

    if current_para:
        cleaned_lines.append(' '.join(current_para))

    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'[ \t]+', ' ', result)
    return result.strip()


def format_docs(docs):
    """Formats retrieved documents into clean, cited context blocks."""
    formatted_blocks = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", 0) + 1
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        block = f"[Source {i} | Page {page} | Chunk #{chunk_id}]:\n{doc.page_content.strip()}"
        formatted_blocks.append(block)
    return "\n\n".join(formatted_blocks)


def main():
    pdf_path = "Alderbridge-Employee-Rulebook.pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"Loading '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Total pages loaded: {len(documents)}")

    # 1. Clean PDF text
    for doc in documents:
        doc.page_content = clean_pdf_text(doc.page_content)
    print("Data cleaning completed successfully across all pages.")

    # 2. Embedding Model (Normalized for Cosine Space)
    print("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 3. Semantic Chunking
    print("Performing Semantic Chunking...")
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    semantic_chunks = semantic_chunker.split_documents(documents)

    for idx, chunk in enumerate(semantic_chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["char_count"] = len(chunk.page_content)

    print(f"Generated {len(semantic_chunks)} semantic chunks.")

    # 4. Local Persistent Vector Database (Chroma)
    db_dir = "./chroma_db"
    collection_name = "alderbridge_employee_rulebook"
    print(f"Connecting to local Chroma database at '{db_dir}'...")

    client = chromadb.PersistentClient(path=db_dir)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=semantic_chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Local database indexed and ready.")

    # 5. Build Retrievers
    print("\nConfiguring retrievers...")
    # Similarity Retriever
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # MMR (Maximal Marginal Relevance) Retriever
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )
    print("Created Similarity and MMR Retrievers.")

    # 6. Test Retrieval
    queries = [
        "What are the standard office hours and break schedules?",
        "What is the policy for sick leave and absence notice?",
        "What are the rules regarding gifts and conflicts of interest?",
        "What is the procedure for resignation and returning company equipment?"
    ]

    for query in queries:
        print("\n" + "=" * 75)
        print(f"QUERY: {query}")
        print("=" * 75)
        docs = mmr_retriever.invoke(query)
        print(format_docs(docs))

    # 7. LCEL RAG Chain
    print("\n" + "=" * 75)
    print("LCEL RAG PROMPT CHAIN TEST")
    print("=" * 75)

    prompt_template = ChatPromptTemplate.from_template("""You are an expert HR assistant answering questions based solely on the official Alderbridge Employee Rulebook.

Official Context:
{context}

Employee Question:
{question}

Answer with concise, policy-compliant instructions and cite the applicable Policy ID or Section:""")

    rag_prompt_chain = (
        {"context": mmr_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
    )

    test_q = "What happens if I need to report a workplace injury or hazard?"
    assembled = rag_prompt_chain.invoke(test_q)
    print(f"\nQuestion: {test_q}\n")
    print("Assembled Context:\n", assembled.to_string()[:400], "...")

    # Optional Gemini LLM invocation if API key is present
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            full_chain = rag_prompt_chain | llm | StrOutputParser()
            response = full_chain.invoke(test_q)
            print("\nGemini Answer:\n", response)
        except Exception as e:
            print("\nLLM call error:", e)
    else:
        print("\n[GOOGLE_API_KEY not set - prompt context generated successfully for offline execution]")


if __name__ == "__main__":
    main()
