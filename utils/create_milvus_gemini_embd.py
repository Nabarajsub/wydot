# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ---- Google GenAI: tuned model (chat) ----
import google.generativeai as genai_chat

# ---- Google GenAI: embeddings (query vectors) ----
from google import genai as genai_embed
from google.genai import types as genai_types

# ---- Milvus (direct) ----
from pymilvus import connections, utility, Collection

# =========================
# ENV + CONFIG
# =========================
load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "19530"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "wydotspec_llamaparse")

CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "./chat_history.sqlite3")

# Tuned Vertex/AI-Studio model + key
TUNED_MODEL_ID = os.getenv("WYDOT_TUNED_MODEL_ID", "tunedModels/wydot-chat-flash-2.5")
WYDOT_FLASH_API_KEY = os.getenv("WYDOT_FLASH_API_KEY")
if not WYDOT_FLASH_API_KEY:
    raise RuntimeError("Missing WYDOT_FLASH_API_KEY in environment.")

# Configure chat model
genai_chat.configure(api_key=WYDOT_FLASH_API_KEY)

# Embedding model config (should match what you used to build the collection)
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Prefer GEMINI/GOOGLE_API_KEY for embeddings; fallback to WYDOT_FLASH_API_KEY if needed
_EMBED_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or WYDOT_FLASH_API_KEY
)

# Vertex option for embeddings (optional)
_USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes", "on"}
_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION") or "us-central1"

# How much recent chat to include each LLM turn
MAX_HISTORY_MSGS = 20
HISTORY_PAIRS_FOR_PROMPT = 6

VECTOR_FIELD = "vector"     # <— your field name from the builder
METRIC_TYPE = "COSINE"      # you used COSINE in your builder script

# =========================
# SQLITE CHAT STORE
# =========================
class ChatHistoryStore:
    """SQLite store for chat history."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id_id ON messages (session_id, id)"
        )
        self._conn.commit()

    def add(self, session_id: str, role: str, content: str, ts: Optional[float] = None):
        if not session_id:
            session_id = "default"
        if ts is None:
            ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, ts)
            )
            self._conn.commit()

    def recent(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not session_id:
            session_id = "default"
        with self._lock:
            cur = self._conn.execute(
                "SELECT role, content, ts FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cur.fetchall()
        rows.reverse()
        return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]

    def clear_session(self, session_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._conn.commit()

@st.cache_resource(show_spinner=False)
def get_chat_store(path: str):
    return ChatHistoryStore(path)

CHAT_DB = get_chat_store(CHAT_DB_PATH)

def add_to_history(session_id: str, role: str, content: str):
    CHAT_DB.add(session_id, role, content)

def get_history_text(session_id: str, max_pairs: int = HISTORY_PAIRS_FOR_PROMPT) -> str:
    limit = min(MAX_HISTORY_MSGS, 2 * max_pairs)
    msgs = CHAT_DB.recent(session_id, limit=limit)
    lines = []
    for m in msgs:
        prefix = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{prefix}: {m['content']}")
    return "\n".join(lines)

# =========================
# Milvus connect + dimension probe
# =========================
@st.cache_resource(show_spinner=False)
def get_milvus_collection(host: str, port: int, collection: str) -> Tuple[Optional[Collection], Optional[int]]:
    try:
        connections.connect(alias="default", host=host, port=str(port))
    except Exception as e:
        st.error(f"Milvus connect error: {e}")
        return None, None

    if not utility.has_collection(collection):
        st.error(f"Milvus collection not found: {collection}")
        return None, None

    col = Collection(collection)
    # load the collection for search
    try:
        col.load()
    except Exception as e:
        st.warning(f"Milvus load() warning: {e}")

    # read vector dim from schema
    dim = None
    for f in col.schema.fields:
        if f.name == VECTOR_FIELD:
            # dim can be in f.params["dim"] or f.dim depending on version
            dim = f.params.get("dim") if getattr(f, "params", None) else None
            if dim is None:
                dim = getattr(f, "dim", None)
            if dim is not None:
                dim = int(dim)
            break

    if dim is None:
        st.error(f"Could not detect vector dimension for field '{VECTOR_FIELD}'.")
        return None, None

    return col, dim

# =========================
# Embedding helper (Gemini)
# =========================
@st.cache_resource(show_spinner=False)
def get_embed_client():
    if _USE_VERTEX:
        if not _PROJECT:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT for Vertex AI embeddings.")
        return genai_embed.Client(vertexai=True, project=_PROJECT, location=_LOCATION)
    return genai_embed.Client(api_key=_EMBED_KEY)

def _extract_values_any(resp) -> List[float]:
    # Attribute-like
    if hasattr(resp, "embedding") and getattr(resp.embedding, "values", None):
        return [float(x) for x in resp.embedding.values]
    if hasattr(resp, "embeddings") and getattr(resp, "embeddings", None):
        e0 = resp.embeddings[0]
        vals = getattr(e0, "values", None)
        if vals is not None:
            return [float(x) for x in vals]
    # Dict-like fallback
    if isinstance(resp, dict):
        if "embedding" in resp and isinstance(resp["embedding"], dict) and "values" in resp["embedding"]:
            return [float(x) for x in resp["embedding"]["values"]]
        if "embeddings" in resp and isinstance(resp["embeddings"], list) and resp["embeddings"]:
            e0 = resp["embeddings"][0]
            if "values" in e0:
                return [float(x) for x in e0["values"]]
    # Last resort: JSON via SDK helper
    try:
        import json as _json
        js = _json.loads(genai_types.to_json(resp))
        return _extract_values_any(js)
    except Exception:
        pass
    raise ValueError("Unexpected embedding response structure (no 'values' found).")

def embed_query_vector(text: str, dim: int) -> List[float]:
    client = get_embed_client()
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            output_dimensionality=dim,
            task_type="RETRIEVAL_QUERY"
        ),
    )
    vals = _extract_values_any(resp)
    # Optional: L2 normalize for cosine
    import math
    n = math.sqrt(sum(v*v for v in vals))
    if n > 0:
        vals = [v / n for v in vals]
    return vals

# =========================
# RAG: RETRIEVAL (direct Milvus)
# =========================
@st.cache_resource(show_spinner=False)
def init_milvus(host: str, port: int, collection: str):
    return get_milvus_collection(host, port, collection)

def milvus_similarity_search(query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    col, dim = init_milvus(HOST, PORT, COLLECTION_NAME)
    if not col or not dim:
        return "", []

    try:
        qv = embed_query_vector(query, dim)
    except Exception as e:
        st.warning(f"[Embed] {e}")
        return "", []

    try:
        res = col.search(
            data=[qv],
            anns_field=VECTOR_FIELD,
            param={"metric_type": METRIC_TYPE, "params": {"ef": 64}},
            limit=k,
            output_fields=["doc_id", "chunk_id", "page", "section", "source", "content"],
        )
    except Exception as e:
        st.warning(f"[Milvus search] {e}")
        return "", []

    chunks = []
    sources = []
    if res and len(res) > 0:
        for hit in res[0]:
            md = hit.entity
            content = md.get("content") or ""
            chunks.append(content)
            sources.append({
                "page": md.get("page"),
                "source": md.get("source"),
                "preview": (content[:300] if content else "")
            })

    context_text = "\n\n".join(chunks)
    return context_text, sources

# =========================
# Prompt composer
# =========================
def _make_parts(
    query: str,
    context_text: str,
    extracted_text: str,
    uploads: Optional[List[Dict[str, Any]]] = None,
    history_text: str = ""
):
    uploads = uploads or []
    parts: List[Dict[str, Any]] = []

    prompt = (
        "You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).\n"
        "Answer the question from the given context. Ensure clarity, brevity, and human-like responses.\n"
        "Context inside double backticks:``{context}``\n"
        "Question inside triple backticks:```{question}```\n"
        "If question is out of scope, answer it based on your role.\n"
        "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
    ).format(
        context=context_text if context_text else extracted_text,
        question=query
    )
    parts.append({"text": prompt})

    if history_text:
        parts.append({"text": f"CHAT HISTORY (recent):\n{history_text}"})
    if context_text:
        parts.append({"text": f"CONTEXT (from vectorstore):\n{context_text}"})
    if extracted_text:
        parts.append({"text": f"UPLOADED DOC/TEXT CONTEXT:\n{extracted_text}"})

    for item in uploads:
        b = item.get("bytes")
        if not b:
            continue
        mime = item.get("mime") or "application/octet-stream"
        parts.append({"inline_data": {"mime_type": mime, "data": b}})

    if query:
        parts.append({"text": f"QUESTION:\n{query}"})

    return parts

# =========================
# LLM call (single tuned model)
# =========================
def _run_tuned_wydot(parts: List[Dict[str, Any]]) -> str:
    """
    Calls your tuned Flash 2.5 model via google.generativeai using WYDOT_FLASH_API_KEY.
    """
    try:
        model = genai_chat.GenerativeModel(TUNED_MODEL_ID)
        resp = model.generate_content(parts)

        text = (getattr(resp, "text", "") or "").strip()
        if text:
            return text

        # Robust fallback if resp.text is empty
        candidates = getattr(resp, "candidates", None)
        if candidates:
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts_list = getattr(content, "parts", None) if content else None
                if parts_list:
                    for p in parts_list:
                        t = getattr(p, "text", None)
                        if t:
                            return t.strip()

        return "I couldn't generate an answer for that."
    except Exception as e:
        return f"[Model error] {e}"

# =========================
# PUBLIC PIPELINE
# =========================
def resultDocuments(
    query: str,
    extracted_text: str = "",
    uploads: Optional[List[Dict[str, Any]]] = None,
    session_id: str = "default",
):
    uploads = uploads or []
    context_text, sources = milvus_similarity_search(query, k=5)

    history_text = get_history_text(session_id, max_pairs=HISTORY_PAIRS_FOR_PROMPT)
    if query:
        add_to_history(session_id, "user", query)

    parts = _make_parts(
        query=query,
        context_text=context_text,
        extracted_text=extracted_text,
        uploads=uploads,
        history_text=history_text,
    )

    answer = _run_tuned_wydot(parts)

    if answer:
        add_to_history(session_id, "assistant", answer)

    return {"text": answer, "sources": sources}

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="WYDOT employee bot", page_icon="🛣️", layout="wide")

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    host = st.text_input("Milvus host", HOST)
    port = st.number_input("Milvus port", min_value=1, max_value=65535, value=PORT, step=1)
    collection = st.text_input("Milvus collection", COLLECTION_NAME)

    if st.button("🔄 Reconnect Milvus"):
        # refresh cached connection/dim tuple
        init_milvus.clear()
        init_milvus(host, port, collection)
        st.success("Reconnected.")

    st.markdown("---")
    st.markdown("## 🤖 Model (fixed)")
    st.write("Using tuned Gemini model:")
    st.code(TUNED_MODEL_ID, language="text")
    st.caption("Configured via WYDOT_FLASH_API_KEY")

    st.markdown("---")
    st.markdown("## 💬 Session")
    default_sid = st.session_state.get("session_id", "default")
    session_id = st.text_input("Session ID", value=default_sid, help="Use a stable ID per user/thread.")
    if st.button("Set session"):
        st.session_state["session_id"] = session_id
        st.success(f"Session set: {session_id}")

    if st.button("🧹 Clear this session history"):
        CHAT_DB.clear_session(session_id)
        st.toast("Session history cleared.", icon="🧹")

    st.markdown("---")
    st.markdown("## 📎 Uploads & Extra Text")
    uploaded_files = st.file_uploader(
        "Optional: attach images/audio/video/docs (sent to the model as raw bytes where supported)",
        accept_multiple_files=True
    )
    extracted_text = st.text_area(
        "Optional: paste OCR/STT/parsed text from uploads to ground answers",
        height=140,
        placeholder="Paste text extracted from a PDF, image, or audio transcript…"
    )

# Main layout
col_chat, col_docs = st.columns([2, 1])

with col_chat:
    st.markdown("### 🛣️ WyDOT employee bot")
    st.markdown('<div id="chat_top"></div>', unsafe_allow_html=True)

    # Render chat history (scrolls with page)
    history_msgs = CHAT_DB.recent(st.session_state.get("session_id", "default"), limit=MAX_HISTORY_MSGS)
    for m in history_msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

    # Chat input
    user_query = st.chat_input("Ask something (e.g., 'PPE for bridge deck pour near traffic') …")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        # Prepare uploads as bytes for the model
        uploads_payload = []
        if uploaded_files:
            for f in uploaded_files:
                try:
                    uploads_payload.append({
                        "bytes": f.getvalue(),
                        "mime": f.type or "application/octet-stream",
                        "name": f.name,
                    })
                except Exception as e:
                    st.warning(f"Failed to read {getattr(f, 'name', 'file')}: {e}")

        with st.spinner("Thinking…"):
            resp = resultDocuments(
                query=user_query,
                extracted_text=extracted_text,
                uploads=uploads_payload,
                session_id=st.session_state.get("session_id", "default"),
            )

        with st.chat_message("assistant"):
            st.write(resp["text"])

        st.session_state["last_sources"] = resp.get("sources", [])

        # Smooth-scroll to the bottom after rendering both bubbles
        st.markdown('<div id="chat_bottom"></div>', unsafe_allow_html=True)
        components.html(
            """<script>
            const el = parent.document.getElementById('chat_bottom');
            if (el) el.scrollIntoView({behavior:'smooth', block:'end'});
            </script>""",
            height=0,
        )

    # Keep a bottom anchor so reruns also land at the latest message
    st.markdown('<div id="chat_bottom"></div>', unsafe_allow_html=True)
    components.html(
        """<script>
        const el = parent.document.getElementById('chat_bottom');
        if (el) el.scrollIntoView({behavior:'smooth', block:'end'});
        </script>""",
        height=0,
    )

with col_docs:
    st.markdown("### 📚 Retrieved Documents")
    st.caption("Top-k chunks retrieved from Milvus for the latest question.")
    sources = st.session_state.get("last_sources", [])
    if not sources:
        st.info("Ask a question to see retrieved documents here.")
    else:
        for i, s in enumerate(sources, start=1):
            label = f"{i}. {s.get('source') or 'unknown'}"
            with st.expander(label, expanded=(i == 1)):
                st.markdown(f"**Page:** {s.get('page', '—')}")
                preview = s.get("preview", "")
                st.write(preview if preview else "_(no preview)_")
