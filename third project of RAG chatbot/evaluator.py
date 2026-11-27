import json
import os
import re
import time
from statistics import mean
from typing import Any, Callable, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ----------------------------------------
# 1. GEMINI EVALUATOR SETUP
# ----------------------------------------

DEFAULT_EVAL_MODEL = os.getenv("GEMINI_EVAL_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

# Approximate USD per 1M tokens for cost estimation when usage metadata is available.
_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "flash-lite": (0.075, 0.30),
    "flash": (0.15, 0.60),
    "pro": (1.25, 10.00),
}

_CLIENT_CACHE: dict[str, genai.Client] = {}


def resolve_api_key(api_key: str | None = None) -> str | None:
    """Resolve a Gemini API key from an explicit argument or environment variables."""
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return key or None


def resolve_eval_model(model: str | None = None) -> str:
    """Resolve the Gemini judge model name."""
    return (
        model
        or os.getenv("GEMINI_EVAL_MODEL")
        or os.getenv("GEMINI_MODEL")
        or "gemini-2.5-flash"
    )


def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Lazily initialize and cache a google.genai Client for the resolved API key."""
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env, "
            "or pass api_key explicitly."
        )
    if resolved_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[resolved_key] = genai.Client(api_key=resolved_key)
    return _CLIENT_CACHE[resolved_key]


def extract_token_cost(ai_message: Any, model_name: str | None = None) -> float:
    """Estimate USD cost from a LangChain AIMessage or usage metadata dict."""
    usage = getattr(ai_message, "usage_metadata", None)
    if usage is None and isinstance(ai_message, dict):
        usage = ai_message
    if not isinstance(usage, dict):
        return 0.0

    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_token_count") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("candidates_token_count") or 0)
    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    normalized = (model_name or "").lower()
    if "flash-lite" in normalized:
        in_rate, out_rate = _MODEL_PRICING_PER_1M["flash-lite"]
    elif "pro" in normalized:
        in_rate, out_rate = _MODEL_PRICING_PER_1M["pro"]
    else:
        in_rate, out_rate = _MODEL_PRICING_PER_1M["flash"]

    cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    return round(cost, 6)


def _extract_doc_id(doc: Any) -> Any:
    """Extract a document/chunk identifier from a dict or LangChain Document."""
    if isinstance(doc, dict):
        if "id" in doc and doc["id"] is not None:
            return doc["id"]
        if "chunk_id" in doc and doc["chunk_id"] is not None:
            return doc["chunk_id"]
        metadata = doc.get("metadata") or {}
        return metadata.get("chunk_id")

    metadata = getattr(doc, "metadata", None) or {}
    if "chunk_id" in metadata and metadata["chunk_id"] is not None:
        return metadata["chunk_id"]
    return getattr(doc, "id", None)


def _extract_doc_text(doc: Any) -> str:
    """Extract text content from a dict or LangChain Document."""
    if isinstance(doc, dict):
        return str(doc.get("text") if "text" in doc else doc.get("page_content", ""))
    return str(getattr(doc, "page_content", ""))


# ----------------------------------------
# 2. RETRIEVAL METRICS
# ----------------------------------------

def hit_rate_at_k(retrieved_ids: Sequence[Any], relevant_ids: Sequence[Any], k: int = 5) -> int:
    """
    Checks whether at least one relevant document
    appears in the top K retrieved documents.
    """
    top_k = retrieved_ids[:k]
    return int(any(doc_id in relevant_ids for doc_id in top_k))


def reciprocal_rank(retrieved_ids: Sequence[Any], relevant_ids: Sequence[Any]) -> float:
    """
    Finds the rank of the first relevant document.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank

    return 0.0


# ----------------------------------------
# 3. REFUSAL DETECTION
# ----------------------------------------

def detect_refusal(answer: str) -> bool:
    """
    Basic refusal detection tailored to document-grounded RAG responses.
    For production, use manual labels or an LLM judge.
    """
    refusal_phrases = [
        "i cannot answer",
        "i can't answer",
        "i don't know",
        "not available in the document",
        "not mentioned in the document",
        "i cannot help with that",
        "i'm unable to answer",
        "not covered in the rulebook",
        "not covered in the provided",
        "not mentioned in the rulebook",
        "not mentioned in the provided",
        "does not contain information",
        "does not contain any information",
        "does not mention",
        "does not cover",
        "does not specify",
        "not specified in the rulebook",
        "not stated in the rulebook",
        "not provided in the context",
        "no information about",
        "cannot find information",
        "could not find information",
    ]

    normalized_answer = (answer or "").lower()
    return any(phrase in normalized_answer for phrase in refusal_phrases)


# ----------------------------------------
# 4. GROUNDEDNESS AND CORRECTNESS
# ----------------------------------------

def _parse_judge_json(raw_text: str) -> dict[str, Any]:
    """Parse JSON returned by the judge LLM, tolerating markdown code fences."""
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(r"^```json\s*|^```\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON must be an object.")
    return parsed


def _clamp_score(value: Any) -> float | None:
    """Convert a score to a float clamped to [0.0, 1.0], preserving None."""
    if value is None:
        return None
    score = float(value)
    return max(0.0, min(1.0, score))


def llm_judge(
    question: str,
    context: str,
    answer: str,
    reference_answer: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Uses Gemini to evaluate:
    1. Groundedness: Is the answer supported by context?
    2. Correctness: Is the answer factually correct?

    Scores are between 0 and 1.
    """
    prompt = f"""
You are an evaluator for a RAG chatbot.

Evaluate the following response.

QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

REFERENCE ANSWER:
{reference_answer if reference_answer else "Not provided"}

CHATBOT ANSWER:
{answer}

Evaluate these two metrics:

1. Groundedness:
Are the claims in the chatbot answer supported by
the retrieved document context?

2. Correctness:
Is the chatbot answer factually correct compared
with the reference answer and reliable knowledge?

If no reference answer is provided, return null
for correctness rather than guessing.

Return ONLY valid JSON in this format:

{{
    "groundedness": 0.0,
    "correctness": null,
    "reason": "Short explanation"
}}

Use scores between 0 and 1.
"""

    eval_client = get_genai_client(api_key)
    eval_model = resolve_eval_model(model)

    response = eval_client.models.generate_content(
        model=eval_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    if not response.text:
        return {
            "groundedness": None,
            "correctness": None,
            "reason": "Empty response from judge model.",
        }

    try:
        parsed = _parse_judge_json(response.text)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "groundedness": None,
            "correctness": None,
            "reason": f"Failed to parse judge JSON: {exc}",
        }

    groundedness = _clamp_score(parsed.get("groundedness"))
    correctness = (
        None if reference_answer is None else _clamp_score(parsed.get("correctness"))
    )

    return {
        "groundedness": groundedness,
        "correctness": correctness,
        "reason": str(parsed.get("reason", "")),
    }


# ----------------------------------------
# 5. COMPLETE EVALUATION FOR ONE QUESTION
# ----------------------------------------

def evaluate_question(
    question: str,
    retrieved_docs: Sequence[Any],
    answer: str,
    relevant_doc_ids: Sequence[Any] | None = None,
    reference_answer: str | None = None,
    k: int = 5,
    cost: float = 0.0,
    delay: float = 0.0,
    was_refused: bool | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    run_judge: bool = True,
) -> dict[str, Any]:
    """Evaluate a single RAG turn across retrieval, refusal, and LLM-judge metrics."""
    retrieved_ids = [_extract_doc_id(doc) for doc in retrieved_docs]
    context = "\n".join(_extract_doc_text(doc) for doc in retrieved_docs)

    # Retrieval metrics (None when ground-truth relevant IDs are not supplied, e.g. live chat)
    if relevant_doc_ids is None:
        hit = None
        mrr = None
    else:
        hit = hit_rate_at_k(retrieved_ids, relevant_doc_ids, k)
        mrr = reciprocal_rank(retrieved_ids, relevant_doc_ids)

    # Refusal metric
    refusal = detect_refusal(answer) if was_refused is None else bool(was_refused)

    # LLM evaluation
    if run_judge:
        judged = llm_judge(
            question,
            context,
            answer,
            reference_answer,
            api_key=api_key,
            model=model,
        )
    else:
        judged = {
            "groundedness": None,
            "correctness": None,
            "reason": "LLM judge skipped.",
        }

    return {
        "question": question,
        "hit_rate_at_k": hit,
        "mrr": mrr,
        "groundedness": judged["groundedness"],
        "correctness": judged["correctness"],
        "refusal": int(refusal),
        "cost": cost,
        "delay_seconds": delay,
        "judge_reason": judged.get("reason", ""),
    }


# ----------------------------------------
# 6. EVALUATE MULTIPLE QUESTIONS
# ----------------------------------------

def calculate_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute average evaluation metrics across a list of evaluation results."""
    def average_metric(key: str) -> float | None:
        values = [r[key] for r in results if r.get(key) is not None]
        return mean(values) if values else None

    total = len(results)

    return {
        "Hit Rate at K": average_metric("hit_rate_at_k"),
        "MRR": average_metric("mrr"),
        "Groundedness": average_metric("groundedness"),
        "Correctness": average_metric("correctness"),
        "Refusal Rate": average_metric("refusal"),
        "Average Cost": average_metric("cost"),
        "Average Delay (seconds)": average_metric("delay_seconds"),
        "Total Questions": total,
    }


# ----------------------------------------
# 7. DISPLAY RESULTS
# ----------------------------------------

def display_results(results: Sequence[dict[str, Any]]) -> None:
    """Pretty-print per-question evaluation results and the final summary."""
    print("\n========== RAG EVALUATION ==========\n")

    for i, result in enumerate(results, start=1):
        print(f"\nQuestion {i}: {result['question']}")
        for key, value in result.items():
            if key != "question":
                print(f"{key}: {value}")

    summary = calculate_summary(results)

    print("\n========== FINAL SUMMARY ==========\n")

    for metric, score in summary.items():
        print(f"{metric}: {score}")


# ----------------------------------------
# 8. ALDERBRIDGE RULEBOOK BENCHMARK SUITE
# ----------------------------------------

ALDERBRIDGE_BENCHMARK: list[dict[str, Any]] = [
    {
        "question": "What are standard office hours and break schedules?",
        "relevant_doc_ids": [7, 8],
        "reference_answer": (
            "Under Policy 02.01 (Page 4), standard office hours are Monday through Friday, "
            "9:00 a.m. to 5:30 p.m. local time, with a 30-minute unpaid meal break."
        ),
    },
    {
        "question": "How many days of paid sick leave do full-time employees get?",
        "relevant_doc_ids": [10, 12],
        "reference_answer": (
            "Under Policy 03.02 (Page 5), full-time employees receive eight paid sick days per "
            "calendar year, with proportional allocations for part-time and midyear hires."
        ),
    },
    {
        "question": "What are the rules on gifts and conflicts of interest?",
        "relevant_doc_ids": [14, 15],
        "reference_answer": (
            "Under Policy 04.03 (Page 6), employees must disclose personal or financial interests "
            "that could affect hiring, purchasing, or customer decisions, cash or cash equivalents "
            "from suppliers or customers are prohibited, and written approval is required before "
            "accepting any business gift worth more than US$50."
        ),
    },
    {
        "question": "Can I work remotely, and how many days per week are allowed?",
        "relevant_doc_ids": [12],
        "reference_answer": (
            "Under Policy 03.04 (Page 5), employees whose duties can be done remotely may request "
            "up to two remote days per workweek with written manager approval."
        ),
    },
    {
        "question": "What is the notice period for resignation and what must be returned?",
        "relevant_doc_ids": [21],
        "reference_answer": (
            "Under Policy 06.03 (Page 8), resignations must be submitted in writing to your manager "
            "and People Operations with at least 14 calendar days' notice where practicable, and "
            "employees must complete handover and return company devices, badges, keys, and records "
            "by the last working day."
        ),
    },
    {
        "question": "Can I bring a pet dog to work?",
        "relevant_doc_ids": None,
        "reference_answer": (
            "The Alderbridge Employee Rulebook does not contain a policy covering bringing pets "
            "or dogs to work."
        ),
    },
]


def run_benchmark_evaluation(
    ask_fn: Callable[[str], tuple[list[Any], str, float]],
    cases: Sequence[dict[str, Any]] | None = None,
    *,
    k: int = 3,
    api_key: str | None = None,
    model: str | None = None,
    run_judge: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run the benchmark suite through a chatbot callable.

    `ask_fn(question)` must return `(retrieved_docs, answer, cost)`.
    """
    dataset = list(cases) if cases is not None else ALDERBRIDGE_BENCHMARK
    results: list[dict[str, Any]] = []
    total = len(dataset)

    for idx, case in enumerate(dataset, start=1):
        question = case["question"]
        if progress_callback:
            progress_callback(idx, total, question)

        start = time.perf_counter()
        retrieved_docs, answer, cost = ask_fn(question)
        delay = round(time.perf_counter() - start, 3)

        result = evaluate_question(
            question=question,
            retrieved_docs=retrieved_docs,
            answer=answer,
            relevant_doc_ids=case.get("relevant_doc_ids"),
            reference_answer=case.get("reference_answer"),
            k=k,
            cost=cost,
            delay=delay,
            api_key=api_key,
            model=model,
            run_judge=run_judge,
        )
        results.append(result)

    return results, calculate_summary(results)


if __name__ == "__main__":
    import sys
    import chromadb
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    db_dir = "./chroma_db"
    collection_name = "alderbridge_employee_rulebook"

    if not os.path.exists(db_dir):
        print(f"Error: '{db_dir}' not found. Run 'python rag_pdf.py' first to build the index.")
        sys.exit(1)

    resolved_key = resolve_api_key()
    chat_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    chroma_client = chromadb.PersistentClient(path=db_dir)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7},
    )

    if not resolved_key:
        print("Error: Set GOOGLE_API_KEY or GEMINI_API_KEY in .env to run full RAG evaluation.")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model=chat_model,
        google_api_key=resolved_key,
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_template(
        """You are an expert HR assistant answering questions based solely on the official Alderbridge Employee Rulebook.

Official Context:
{context}

Employee Question:
{question}

Answer strictly based on the context above. If the information is not covered in the rulebook, clearly state that. Cite the applicable Policy ID or Section:"""
    )
    parser = StrOutputParser()

    def _ask(question: str) -> tuple[list[Any], str, float]:
        docs = retriever.invoke(question)
        context_str = "\n\n".join(
            f"[Page {d.metadata.get('page', 0) + 1} | Chunk #{d.metadata.get('chunk_id', 'N/A')}]:\n{d.page_content.strip()}"
            for d in docs
        )
        ai_msg = llm.invoke(prompt.invoke({"context": context_str, "question": question}))
        answer_text = parser.invoke(ai_msg)
        cost = extract_token_cost(ai_msg, chat_model)
        return docs, answer_text, cost

    eval_results, _ = run_benchmark_evaluation(_ask, k=3, api_key=resolved_key)
    display_results(eval_results)
