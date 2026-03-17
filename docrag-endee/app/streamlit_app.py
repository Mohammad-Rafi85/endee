import streamlit as st
import requests
import json
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DocRAG — Chat with Your Documents",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #e2e8f0; }
.chat-bubble-user {
    background: #1e40af; padding: 12px 16px; border-radius: 12px 12px 2px 12px;
    margin: 8px 0; max-width: 75%; margin-left: auto; color: white;
}
.chat-bubble-bot {
    background: #1e293b; padding: 12px 16px; border-radius: 12px 12px 12px 2px;
    margin: 8px 0; max-width: 75%; color: #e2e8f0; border: 1px solid #334155;
}
.source-card {
    background: #1e293b; border: 1px solid #475569; border-radius: 8px;
    padding: 10px 14px; margin: 4px 0; font-size: 0.85em; color: #94a3b8;
}
.metric-box {
    background: #1e293b; border-radius: 8px; padding: 12px;
    border: 1px solid #334155; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://endee.io/favicon.ico", width=32)
    st.title("DocRAG")
    st.caption("Powered by Endee Vector DB")
    st.divider()

    st.subheader("📄 Upload Document")
    uploaded = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    doc_name = st.text_input("Document name (optional)")

    if st.button("🚀 Index Document", use_container_width=True, type="primary"):
        if uploaded:
            with st.spinner("Indexing…"):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                data = {"doc_name": doc_name or uploaded.name}
                try:
                    r = requests.post(f"{API_BASE}/ingest", files=files, data=data, timeout=60)
                    if r.ok:
                        st.success(f"✅ Indexed {r.json().get('chunks', '?')} chunks!")
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(f"API error: {e}")
        else:
            st.warning("Please upload a file first.")

    st.divider()

    st.subheader("⚙️ Settings")
    top_k = st.slider("Retrieved chunks (top-k)", 1, 10, 4)
    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()

    if st.button("🗑️ Clear index", use_container_width=True):
        try:
            r = requests.delete(f"{API_BASE}/index", timeout=10)
            st.success("Index cleared!" if r.ok else r.text)
        except Exception as e:
            st.error(str(e))

    # Stats
    st.divider()
    st.subheader("📊 Index Stats")
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=5).json()
        col1, col2 = st.columns(2)
        col1.metric("Vectors", stats.get("vector_count", "—"))
        col2.metric("Docs", stats.get("doc_count", "—"))
    except Exception:
        st.caption("Server not reachable")

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🧠 Chat with Your Documents")
st.caption("Upload a document in the sidebar, then ask anything about it.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
    if msg["role"] == "assistant" and show_sources and msg.get("sources"):
        with st.expander(f"📎 {len(msg['sources'])} source chunks"):
            for s in msg["sources"]:
                st.markdown(
                    f'<div class="source-card"><b>{s["doc"]}</b> · chunk {s["chunk_id"]}'
                    f'<br>{s["text"][:300]}…</div>',
                    unsafe_allow_html=True,
                )

# Input
query = st.chat_input("Ask a question about your documents…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="chat-bubble-user">{query}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking…"):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"question": query, "top_k": top_k},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                answer = data.get("answer", "No answer.")
                sources = data.get("sources", [])
            else:
                answer = f"❌ API error: {resp.text}"
                sources = []
        except Exception as e:
            answer = f"❌ Could not reach backend: {e}"
            sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.markdown(f'<div class="chat-bubble-bot">{answer}</div>', unsafe_allow_html=True)

    if show_sources and sources:
        with st.expander(f"📎 {len(sources)} source chunks"):
            for s in sources:
                st.markdown(
                    f'<div class="source-card"><b>{s["doc"]}</b> · chunk {s["chunk_id"]}'
                    f'<br>{s["text"][:300]}…</div>',
                    unsafe_allow_html=True,
                )
