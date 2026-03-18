"""
DocRAG Backend — FastAPI + Endee Python SDK + Sentence Transformers + Gemini/OpenAI
Uses the official Endee Python SDK exactly as documented.
"""
from __future__ import annotations

import io, os, re, uuid
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
ENDEE_URL      = os.getenv("ENDEE_URL", "http://endee:8080")
ENDEE_TOKEN    = os.getenv("ENDEE_TOKEN", "")
INDEX_NAME     = "docrag"
EMBED_DIM      = 384
CHUNK_SIZE     = 400
CHUNK_OVERLAP  = 60
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Endee SDK client ──────────────────────────────────────────────────────────
_client = None
_index  = None

def get_client():
    global _client
    if _client is None:
        from endee import Endee
        # IMPORTANT: The SDK appends /api/v1 automatically. 
        # Set ENDEE_URL to "http://endee:8080" in docker-compose.
        _client = Endee(ENDEE_TOKEN if ENDEE_TOKEN else "")
        _client.set_base_url(ENDEE_URL) 
    return _client

def get_index():
    global _index
    if _index is None:
        _index = get_client().get_index(INDEX_NAME)
    return _index

def ensure_index():
    """Create the Endee index using the official SDK."""
    client = get_client()
    try:
        # Check if index exists using the SDK
        indices = client.list_indexes()
        names = [i.name for i in indices] if indices else []
        
        if INDEX_NAME not in names:
            client.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric_type="cosine" # Use 'metric_type' for compatibility
            )
            print(f"✅ Successfully created index: {INDEX_NAME}")
    except Exception as e:
        print(f"⚠️ Index check/creation failed: {e}")

    if INDEX_NAME not in names:
        try:
            from endee import Precision
            client.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
        except Exception as e:
            if "400" in str(e) or "exist" in str(e).lower() or "already" in str(e).lower():
                pass
            else:
                print(f"[WARN] Index creation: {e}")

# ── Embedder ──────────────────────────────────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def embed(texts: list) -> list:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()

# ── Text utils ────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]

def extract_text(file: UploadFile) -> str:
    raw = file.file.read()
    if file.filename.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise HTTPException(400, "pypdf not installed; only .txt supported")
    return raw.decode("utf-8", errors="ignore")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="DocRAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_doc_store: dict = {}

@app.on_event("startup")
async def startup():
    ensure_index()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.post("/ingest")
async def ingest(file: UploadFile = File(...),
                 doc_name: Optional[str] = Form(None)):
    doc_label = doc_name or file.filename or "document"
    text      = extract_text(file)
    chunks    = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No text chunks found.")

    embeddings = embed(chunks)

    # Build plain dicts exactly as shown in official Endee docs
    vectors = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        cid = str(uuid.uuid4())
        _doc_store[cid] = {"text": chunk, "doc": doc_label, "chunk_id": i}
        vectors.append({
            "id":     cid,
            "vector": vec,
            "meta":   {"doc": doc_label, "chunk_index": i, "text": chunk[:200]},
        })

    # Call upsert exactly as in official docs: index.upsert([{...}, {...}])
    get_index().upsert(vectors)
    return {"status": "ok", "doc": doc_label, "chunks": len(chunks)}

@app.post("/query")
async def query(req: QueryRequest):
    q_vec   = embed([req.question])[0]
    results = get_index().query(vector=q_vec, top_k=req.top_k)

    sources, context_parts = [], []
    for h in results:
        if isinstance(h, dict):
            cid, score, meta = h.get("id",""), h.get("similarity",0), h.get("meta",{}) or {}
        else:
            cid   = getattr(h, "id", "")
            score = getattr(h, "similarity", 0)
            meta  = getattr(h, "meta", {}) or {}
            if not isinstance(meta, dict):
                meta = {}

        stored = _doc_store.get(cid) or {
            "text":     meta.get("text",""),
            "doc":      meta.get("doc","unknown"),
            "chunk_id": meta.get("chunk_index",0),
        }
        sources.append({"doc": stored["doc"], "chunk_id": stored["chunk_id"],
                        "text": stored["text"], "score": round(float(score),4)})
        context_parts.append(stored["text"])

    context = "\n\n---\n\n".join(context_parts)
    answer  = await generate_answer(req.question, context)
    return {"answer": answer, "sources": sources}

async def generate_answer(question: str, context: str) -> str:
    # Pull the key directly inside the function to ensure it's not empty
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if gemini_key:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-1.5-flash:generateContent?key={gemini_key}",
                    json={"contents":[{"parts":[{"text":(
                        "Answer ONLY using the document context. "
                        "If not found, say so.\n\n"
                        f"Context:\n{context}\n\nQuestion: {question}"
                    )}]}],
                    "generationConfig":{"maxOutputTokens":512,"temperature":0.2}},
                )
            if r.is_success:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                # This helps you debug in the terminal logs
                print(f"Gemini API Error: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"Gemini Connection Error: {e}")

    # ... rest of your code for OpenAI and Fallback ...

    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"model":"gpt-4o-mini","messages":[
                        {"role":"system","content":"Answer ONLY from context."},
                        {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"},
                    ],"max_tokens":512,"temperature":0.2},
                )
            if r.is_success:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    if context:
        sentences = re.split(r"(?<=[.!?])\s+", context)
        q_words   = set(question.lower().split())
        best      = max(sentences, key=lambda s: len(q_words & set(s.lower().split())))
        return (f"Based on the document: {best.strip()}\n\n"
                f"*(Add GEMINI_API_KEY in docker-compose.yml for full AI answers.)*")
    return "I couldn't find relevant information in the indexed documents."

@app.get("/stats")
def stats():
    try:
        info  = get_index().describe()
        count = info.get("vector_count", len(_doc_store)) if isinstance(info,dict) \
                else getattr(info,"vector_count",len(_doc_store))
        return {"vector_count": count,
                "doc_count": len({v["doc"] for v in _doc_store.values()})}
    except Exception:
        return {"vector_count": len(_doc_store), "doc_count": 0}

@app.delete("/index")
def clear_index():
    global _index
    try:
        get_client().delete_index(INDEX_NAME)
        _doc_store.clear()
        _index = None
        ensure_index()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
