# 🎓 PDF Tutor AI — Lightweight Hybrid RAG Chatbot

An educational local chatbot that processes textbooks and academic PDFs, explains concepts, answers student questions, generates self-assessment quizzes, and supports follow-up questions using **Hybrid Retrieval Augmented Generation (RAG)**.

Designed to run **100% locally on standard Mac/PC CPUs** with **zero local neural networks, zero Ollama, and zero heavy transformer model downloads**.

---

## 🌟 Key Features

- **Lightweight Architecture:** No local LLMs, no Hugging Face model downloads, no GPU/CUDA requirements, and no local neural inference.
- **Page-Aware PDF Ingestion:** Uses PyMuPDF to extract text page-by-page, preserving accurate page citations and preventing metadata bleed across pages.
- **Scanned PDF Detection:** Identifies scanned, image-only documents and alerts the user that OCR is not supported.
- **Lexical Vector Retrieval (TF-IDF):** Fits scikit-learn `TfidfVectorizer` and calculates Cosine Similarity over document chunks.
- **ChromaDB Persistent Storage:** Persists vector representations and page metadata to disk without downloading neural embedding functions.
- **Sparse Keyword Retrieval (BM25):** Employs `rank-bm25` (BM25Okapi) for precise technical terminology, acronyms, and exact phrase lookups.
- **Reciprocal Rank Fusion (RRF):** Fuses dense vector rankings with sparse keyword rankings using the formula:
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)} \quad (k = 60)$$
- **Dual Answer Modes:**
  - **Mode A (Gemini Connected):** Natural language tutoring using Google Gemini API (`gemini-2.5-flash`, `gemini-1.5-flash`, or `gemini-1.5-pro`) following strict pedagogical rules and citations.
  - **Mode B (Retrieval-Only):** Completely free mode that displays retrieved passages, exact page numbers, and relevance scores without requiring any API key.
- **7 Tutoring Learning Modes:**
  1. `Explain Concept`
  2. `Explain Like I'm a Beginner` (ELI5)
  3. `Detailed Explanation`
  4. `Summarize`
  5. `Generate Quiz`
  6. `Test My Knowledge`
  7. `Ask Follow-up Question`
- **Security & Prompt Injection Protection:** Treats retrieved PDF passages as untrusted input enclosed within strict security boundary tags to prevent prompt injection.

---

## 📐 Architecture

```
User (Browser)
      │
      ▼
Streamlit Frontend (app.py)
      │
      ├── Ingestion Pipeline
      │     ├── PyMuPDF (Page extraction & validation)
      │     └── RecursiveCharacterTextSplitter (Page-aware chunking: 800 chars / 150 overlap)
      │
      ├── Hybrid Indexing & Retrieval
      │     ├── TF-IDF Vectorizer (scikit-learn) + Cosine Similarity
      │     ├── ChromaDB (Persistent storage with embedding_function=None)
      │     ├── BM25Okapi (rank-bm25 keyword index)
      │     └── Reciprocal Rank Fusion (RRF with k=60)
      │
      └── Tutoring Engine
            ├── Mode A: Google Gemini API (gemini-2.5-flash / official google-genai SDK)
            └── Mode B: Fallback Retrieval-Only Mode (Zero external dependencies)
```

---

## 🚀 Installation & Setup (macOS)

### 1. Prerequisites
Ensure you have Python 3.12 installed on your Mac:
```bash
brew install python@3.12
```

### 2. Clone / Navigate to the Project
```bash
cd "Basic Tutor Chatbot"
```

### 3. Create & Activate Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Optional Gemini API Configuration

To enable **Mode A** (Natural-Language Answer Generation):

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste your key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
*(Alternatively, you can paste your API key directly into the Streamlit sidebar at runtime.)*

> **Note:** If no API key is provided, the application runs automatically in **Retrieval-Only Mode (Mode B)** with full search and citation capabilities!

---

## 🏃 Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default browser at:
`http://localhost:8501`

### How to Use:
1. Enter the absolute path to any text-searchable PDF in the sidebar:
   ```
   /Users/username/Documents/OperatingSystems.pdf
   ```
2. Click **Load PDF**.
3. Select your desired **Tutoring Mode** (e.g. *Explain Concept*, *Summarize*, or *Generate Quiz*).
4. Ask any question in the chat input.
5. Review the answer, verify the **Page Citations**, and click **Inspect Retrieved Passages** to review the exact source chunks with their RRF, TF-IDF, and BM25 scores.

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest -v
```

All 23+ tests verify:
- Valid PDF loading and page extraction
- Invalid path, empty PDF, and corrupted PDF handling
- Scanned PDF detection (OCR notice)
- Page-aware chunking and metadata boundary isolation
- TF-IDF vector generation, similarity ranking, and caching
- BM25 index construction and exact term retrieval
- RRF calculation, deduplication, and ranking
- ChromaDB persistence and synchronization
- Tutor engine context building and citation accuracy
- Retrieval-only fallback when API key is absent
- Prompt injection defense delimiters

---

## ⚠️ Known Limitations

1. **OCR Not Supported in v1:** Image-only scanned PDFs without embedded text layers cannot be read. The application detects these and prompts the user to supply a searchable PDF.
2. **Lexical Vector Retrieval:** TF-IDF captures exact n-grams and term statistics. It does not model deep semantic synonyms the way high-resource neural embeddings do, but it requires 0 MB of GPU/neural model memory.
3. **Complex Tables & Figures:** Tabular data and embedded images are extracted as plain text streams; complex multi-column layouts may have linearized text.

---

## 📄 License
MIT License. Built for educational and learning purposes.
