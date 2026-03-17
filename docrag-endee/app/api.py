"""
DocRAG Backend — FastAPI + Endee Vector DB + Sentence Transformers + OpenAI
"""
from __future__ import annotations

import io
import os
import re
import textwrap
import uuid
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")
INDEX_NAME = "docrag"
EMBED_DIM = 384          # sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE = 400         # chars
CHUNK_OVERLAP = 60

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")   # optional; falls back to extractive

# ── Lazy imports ──────────────────────────────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── Endee helpers ─────────────────────────────────────────────────────────────
def endee_headers():
    h = {"Content-Type": "application/json"}
    if ENDEE_TOKEN:
        h["Authorization"] = ENDEE_TOKEN
    return h


def endee_get(path: str):
    with httpx.Client(timeout=15) as c:
        r = c.get(f"{ENDEE_URL}{path}", headers=endee_headers())
    r.raise_for_status()
    return r.json()


def endee_post(path: str, body: dict):
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{ENDEE_URL}{path}", json=body, headers=endee_headers())
    r.raise_for_status()
    return r.json()


def endee_delete(path: str):
    with httpx.Client(timeout=15) as c:
        r = c.delete(f"{ENDEE_URL}{path}", headers=endee_headers())
    r.raise_for_status()
    return r.json()


def ensure_index():
    """Create the Endee index if it doesn't exist yet."""
    try:
        existing = endee_get("/index/list")
        names = [i["name"] for i in existing.get("indexes", [])]
        if INDEX_NAME not in names:
            endee_post("/index/create", {
                "name": INDEX_NAME,
                "dimension": EMBED_DIM,
                "space_type": "cosine",
                "precision": "int8",
            })
    except Exception as e:
        raise RuntimeError(f"Could not ensure Endee index: {e}")


def upsert_vectors(items: list[dict]):
    endee_post(f"/index/{INDEX_NAME}/upsert", {"vectors": items})


def query_vectors(vector: list[float], top_k: int):
    result = endee_post(f"/index/{INDEX_NAME}/query", {
        "vector": vector,
        "top_k": top_k,
    })
    return result.get("results", [])


# ── Text utils ────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def extract_text_from_upload(file: UploadFile) -> str:
    raw = file.file.read()
    if file.filename.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise HTTPException(400, "pypdf not installed; only .txt supported")
    return raw.decode("utf-8", errors="ignore")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="DocRAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_doc_store: dict[str, dict] = {}   # chunk_id → {text, doc, chunk_id}


@app.on_event("startup")
async def startup():
    ensure_index()


# ── Routes ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    doc_name: Optional[str] = Form(None),
):
    doc_label = doc_name or file.filename or "document"
    text = extract_text_from_upload(file)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Document produced no text chunks.")

    vectors_payload = []
    embeddings = embed(chunks)
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        cid = str(uuid.uuid4())
        _doc_store[cid] = {"text": chunk, "doc": doc_label, "chunk_id": i}
        vectors_payload.append({
            "id": cid,
            "vector": vec,
            "meta": {"doc": doc_label, "chunk_index": i, "text": chunk[:200]},
        })

    upsert_vectors(vectors_payload)
    return {"status": "ok", "doc": doc_label, "chunks": len(chunks)}


@app.post("/query")
async def query(req: QueryRequest):
    q_vec = embed([req.question])[0]
    hits = query_vectors(q_vec, req.top_k)

    sources = []
    context_parts = []
    for h in hits:
        cid = h.get("id", "")
        stored = _doc_store.get(cid) or {
            "text": h.get("meta", {}).get("text", ""),
            "doc": h.get("meta", {}).get("doc", "unknown"),
            "chunk_id": h.get("meta", {}).get("chunk_index", 0),
        }
        sources.append({
            "doc": stored["doc"],
            "chunk_id": stored["chunk_id"],
            "text": stored["text"],
            "score": round(h.get("similarity", 0), 4),
        })
        context_parts.append(stored["text"])

    context = "\n\n---\n\n".join(context_parts)
    answer = await generate_answer(req.question, context)
    return {"answer": answer, "sources": sources}


async def generate_answer(question: str, context: str) -> str:
    """Use OpenAI if key is set, else fall back to extractive answer."""
    if OPENAI_API_KEY:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. Answer the user's question "
                                "using ONLY the provided document context. If the answer "
                                "is not in the context, say so clearly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}",
                        },
                    ],
                    "max_tokens": 512,
                    "temperature": 0.2,
                },
            )
        if r.is_success:
            return r.json()["choices"][0]["message"]["content"].strip()

    # Extractive fallback — return most relevant sentence from top chunk
    if context:
        sentences = re.split(r"(?<=[.!?])\s+", context)
        q_words = set(question.lower().split())
        best = max(sentences, key=lambda s: len(q_words & set(s.lower().split())))
        return (
            f"Based on the document: {best.strip()}\n\n"
            f"*(Set OPENAI_API_KEY for a full generative answer.)*"
        )
    return "I couldn't find relevant information in the indexed documents."


@app.get("/stats")
def stats():
    try:
        info = endee_get(f"/index/{INDEX_NAME}/info")
        return {
            "vector_count": info.get("vector_count", len(_doc_store)),
            "doc_count": len({v["doc"] for v in _doc_store.values()}),
        }
    except Exception:
        return {"vector_count": len(_doc_store), "doc_count": 0}


@app.delete("/index")
def clear_index():
    try:
        endee_delete(f"/index/{INDEX_NAME}/vectors")
        _doc_store.clear()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
