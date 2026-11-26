# Multiverse™ 3D Spatial Platform & Autonomous RAG Chatbot

An enterprise-grade, immersive 3D spatial web platform and Retrieval-Augmented Generation (RAG) chatbot engine powered by a custom 7-stage document ingestion architecture:

$$\text{Upload} \longrightarrow \text{Load} \longrightarrow \text{Split} \longrightarrow \text{Clean} \longrightarrow \text{Chunk} \longrightarrow \text{Embedding} \longrightarrow \text{Database}$$

---

## 🌟 Key Features

1. **Complete RAG System**:
   - Custom **Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping (HC-SW Chunking)**.
   - Dense 384-dimensional vector embeddings with `all-MiniLM-L6-v2`.
   - Local, persistent **ChromaDB** vector database with cosine similarity search.
   - Grounded context synthesis with source breadcrumbs, verbatim citations, and confidence scores.

2. **3D Interactive Web Platform**:
   - **Real-Time Three.js WebGL 3D Universe**: An interactive 3D quantum nodal network where users can orbit, zoom, rotate parallel world nodes, and observe entangled particle streams.
   - **120 FPS Sub-Frame Scrollytelling 3D Engine**: Cinematic 20-frame camera traversal through Einstein-Rosen wormholes and connected dimensional realities (all frames cleaned of watermarks and logos).
   - **Quantum Superposition Resonance HUD**: Dynamic canvas visualizer showing waveform coherence.

3. **Multiverse™ Intelligence Chatbot**:
   - Answers any technical, architectural, or operational question about **Multiverse™** (Helios-Q9 quantum processor, DPTP wormholes, 32K photonic waveguide displays, acoustic levitation haptics, and multi-agent AI).
   - Available both as an always-present **Floating Holographic 3D Chatbot** and a full-screen **RAG Intelligence Studio**.
   - Supports uploading custom reality specifications or markdown/text documents to trigger the live 7-stage ingestion pipeline.

---

## 🚀 Quick Start

### 1. Launch the 3D Website & RAG Server
Run the one-click launcher:
```bash
./start.sh
```
Or directly with Python:
```bash
.venv/bin/python server.py 8080
```
Then open your browser and navigate to:
👉 **http://localhost:8080**

### 2. Interactive Terminal Chatbot
You can also chat with the Multiverse RAG knowledge base directly in your terminal:
```bash
.venv/bin/python chatbot.py
```
Or query a single question directly:
```bash
.venv/bin/python chatbot.py "What is Multiverse?"
```

### 3. Run Ingestion Pipeline
To re-run the 7-stage ingestion pipeline on the Multiverse manual:
```bash
.venv/bin/python pipeline.py
```

### 4. Run Automated Unit Tests
```bash
.venv/bin/python -m unittest test_pipeline.py
```

---

## 🔬 Chunking Methodology (HC-SW Chunking)

- **Name**: Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping
- **Sentence Boundary Snapping**: Greedy accumulation of atomic sentences snapping cleanly at sentence endings without chopping mid-thought.
- **Hierarchical Breadcrumbs**: Injects document breadcrumbs (`[Context: Multiverse > Section > Subsection]`) into the chunk text so the embedding model encodes structural ancestry and local details.
- **Sliding Overlap**: Contextual trailing sentence overlap providing seamless continuity across chunk transitions.

---

## 📂 Project Structure

```
RAG Project Second Chatbot/
├── chatbot.py                 # CLI Interactive Chatbot
├── data/
│   ├── multiverse_manual.md   # Primary Multiverse Technical Documentation
│   └── ai_agent_systems.md    # Autonomous AI Agent Systems Reference
├── pipeline.py                # Standalone 7-stage Ingestion Pipeline CLI
├── requirements.txt           # Python dependencies
├── server.py                  # Production HTTP Web Server & RAG API
├── src/
│   ├── chunker.py             # Custom HC-SW Chunking Engine
│   ├── cleaner.py             # Text Normalization & Cleaning
│   ├── database.py            # ChromaDB Vector Database Interface
│   ├── embedder.py            # Sentence-Transformers Vector Embedder
│   ├── loader.py              # Encoding Detection & Metrics Loader
│   ├── models.py              # Pydantic / Dataclass Models
│   ├── pipeline_runner.py     # Dynamic End-to-End Orchestrator
│   ├── rag_engine.py          # Grounded Context Retrieval & Synthesis
│   ├── splitter.py            # Structural Markdown Splitter
│   └── uploader.py            # Cryptographic Validation & Staging
├── start.sh                   # One-click startup shell script
├── static/
│   ├── frames/                # 20 High-Res Inpainted 3D Journey Frames
│   └── index.html             # 3D WebGL Multiverse Website & RAG Studio
├── test_pipeline.py           # Comprehensive Unit Tests
└── web_chatbot.py             # Web Server Entrypoint
```
