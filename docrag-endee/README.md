# DocRAG — Chat with Your Documents using Endee Vector DB

> **A full-stack RAG (Retrieval Augmented Generation) application** that lets you upload any document and ask questions about it in natural language — powered by **[Endee](https://github.com/endee-io/endee)** as the vector database.

---

## 📸 Preview

```
┌─────────────────────────────────────────────────────────────────┐
│  Sidebar                │  Chat                                  │
│  ─────────────────────  │  ─────────────────────────────────    │
│  📄 Upload Document     │  You: What is RAG?                    │
│  [Upload .txt / .pdf]   │                                        │
│  [🚀 Index Document]    │  Bot: RAG (Retrieval Augmented         │
│                         │  Generation) combines a retrieval      │
│  ⚙️  Settings           │  system with a generative model...    │
│  top-k slider           │                                        │
│  show sources toggle    │  📎 3 source chunks ▸                  │
│                         │                                        │
│  📊 Index Stats         │  You: What is Endee?                  │
│  Vectors: 42  Docs: 1   │  …                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ System Design

```
┌──────────────┐    upload     ┌──────────────┐   upsert vectors  ┌──────────────────┐
│  Streamlit   │ ────────────► │  FastAPI     │ ────────────────► │  Endee Vector DB │
│  Frontend    │               │  Backend     │                    │  (port 8080)     │
│  (port 8501) │ ◄──────────── │  (port 8000) │ ◄──────────────── │                  │
└──────────────┘    answer +   └──────────────┘   top-k results   └──────────────────┘
                    sources          │
                                     │ embed (local)
                              ┌──────▼──────────┐
                              │ sentence-        │
                              │ transformers     │
                              │ all-MiniLM-L6-v2 │
                              └─────────────────┘
                                     │ generate answer (optional)
                              ┌──────▼──────────┐
                              │  OpenAI          │
                              │  GPT-4o-mini     │
                              │  (optional)      │
                              └─────────────────┘
```

### How it works

1. **Ingest** — User uploads a `.txt` or `.pdf` file via the Streamlit UI.
2. **Chunk** — Backend splits the document into overlapping text chunks (~400 chars).
3. **Embed** — Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, runs locally).
4. **Store** — Embeddings are upserted into **Endee** under the `docrag` index using cosine similarity.
5. **Query** — User asks a question; it's embedded and the top-k most similar chunks are retrieved from Endee.
6. **Generate** — An LLM (GPT-4o-mini if key provided, else extractive fallback) crafts a grounded answer from the retrieved chunks.
7. **Display** — Answer and source chunks are shown in the Streamlit chat UI.

---

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose v2
- 2 GB RAM minimum

### 1. Star & Fork (required by assignment)

```bash
# Star the repo at https://github.com/endee-io/endee
# Then fork it and clone your fork:
git clone https://github.com/<your-username>/endee
```

### 2. Clone this project

```bash
git clone https://github.com/<your-username>/docrag-endee
cd docrag-endee
```

### 3. (Optional) Add your OpenAI key

Edit `docker-compose.yml` and set:
```yaml
OPENAI_API_KEY: "sk-..."
```

Without it, the app uses an **extractive fallback** — it returns the most relevant sentence from the retrieved chunks. The semantic search via Endee works in both cases.

### 4. Start everything

```bash
docker compose up --build
```

Services started:
| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI API | http://localhost:8000 |
| Endee Dashboard | http://localhost:8080 |

### 5. Index the sample document

```bash
# From another terminal (while docker compose is running):
pip install requests
python scripts/ingest_sample.py
```

Or upload any `.txt` / `.pdf` directly via the sidebar in the UI.

### 6. Ask questions!

Open http://localhost:8501 and start chatting.

---

## 🖥️ Running without Docker (local dev)

```bash
# 1. Start Endee
docker run -p 8080:8080 -v endee-data:/data endeeio/endee-server:latest

# 2. Install Python deps
pip install -r requirements.txt

# 3. Start the API
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the UI (new terminal)
streamlit run app/streamlit_app.py

# 5. Ingest sample data
python scripts/ingest_sample.py
```

---

## 🔍 Use of Endee

This project uses Endee as the **sole vector storage and retrieval engine**.

| Operation | Endee API endpoint |
|---|---|
| Create index | `POST /api/v1/index/create` |
| Store embeddings | `POST /api/v1/index/{name}/upsert` |
| Semantic search | `POST /api/v1/index/{name}/query` |
| Index statistics | `GET /api/v1/index/{name}/info` |
| Clear all vectors | `DELETE /api/v1/index/{name}/vectors` |

**Index configuration:**
```json
{
  "name": "docrag",
  "dimension": 384,
  "space_type": "cosine",
  "precision": "int8"
}
```

Endee's HNSW-based ANN indexing with INT8 quantization delivers fast retrieval even for large document collections — making it ideal for RAG workloads.

---

## 📁 Project Structure

```
docrag-endee/
├── app/
│   ├── api.py               # FastAPI backend (chunking, embedding, Endee calls)
│   └── streamlit_app.py     # Streamlit chat frontend
├── data/
│   └── sample_knowledge_base.txt   # Sample AI/ML knowledge base for demo
├── scripts/
│   └── ingest_sample.py     # CLI script to quickly ingest sample data
├── docker-compose.yml       # Orchestrates Endee + API + UI
├── Dockerfile.api           # FastAPI container
├── Dockerfile.ui            # Streamlit container
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `ENDEE_URL` | `http://localhost:8080/api/v1` | Endee server URL |
| `ENDEE_TOKEN` | `""` | Auth token (match `NDD_AUTH_TOKEN`) |
| `OPENAI_API_KEY` | `""` | OpenAI key for GPT-4o-mini answers |
| `API_BASE_URL` | `http://localhost:8000` | Backend URL (for Streamlit) |

---

## 🧪 Sample Questions

After ingesting `sample_knowledge_base.txt`, try:

- *"What is RAG and how does it work?"*
- *"What is the difference between supervised and unsupervised learning?"*
- *"How does Endee compare to other vector databases?"*
- *"What are transformers in NLP?"*
- *"How do you prevent overfitting in machine learning?"*

---

## 📄 License

Apache 2.0 — same as Endee itself.

---

## 🙏 Credits

- **[Endee](https://github.com/endee-io/endee)** — vector database
- **[sentence-transformers](https://www.sbert.net/)** — local embeddings
- **[FastAPI](https://fastapi.tiangolo.com/)** — backend framework
- **[Streamlit](https://streamlit.io/)** — frontend UI
- **OpenAI GPT-4o-mini** — optional answer generation
