# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# ---- RAG libs ----
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.milvus import Milvus

# ---- LLM Providers (optional imports) ----
import google.generativeai as genai

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

try:
    from groq import Groq
    _HAS_GROQ = True
except Exception:
    _HAS_GROQ = False
    Groq = None  # type: ignore


# =========================
# ENV + CONFIG
# =========================
load_dotenv()

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "19530"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "wyodotspecs")

# Default Gemini model
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "./chat_history.sqlite3")

# API keys
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# How much recent chat to include each LLM turn
MAX_HISTORY_MSGS = 20
HISTORY_PAIRS_FOR_PROMPT = 6


# =========================
# MODEL ROUTER
# =========================
# Map the user-friendly labels to (provider, actual_model_id)
MODEL_CATALOG = {
    "Gemini 2.5 Pro":        ("gemini", "gemini-2.5-pro"),
    "Gemini 2.5 Flash":      ("gemini", "gemini-2.5-flash"),
    "Gemini 2.5 Flash-Lite": ("gemini", "gemini-2.5-flash-lite"),

    "GPT-5":         ("openai", "gpt-5"),
    "GPT-5 mini":    ("openai", "gpt-5-mini"),
    # user wrote "04-mini"; normalize to "o4-mini"
    "o4-mini":       ("openai", "o4-mini"),
    "GPT-4.1 mini":  ("openai", "gpt-4.1-mini"),

    "llama-3.3-70b-versatile": ("groq", "llama-3.3-70b-versatile"),
}

# Back-compat aliases (if someone sets the env default to old style names)
ALIAS_NORMALIZE = {
    "04-mini": "o4-mini",
    "gpt-4.1 mini": "gpt-4.1-mini",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
}


def resolve_model(label_or_id: str) -> Tuple[str, str]:
    """
    Return (provider, model_id) based on a UI label or raw model id.
    """
    key = label_or_id.strip()
    key = ALIAS_NORMALIZE.get(key, key)

    if key in MODEL_CATALOG:
        return MODEL_CATALOG[key]

    # If a raw Gemini id is passed
    if key.startswith("gemini-"):
        return ("gemini", key)
    # OpenAI quick heuristic
    if key.startswith(("gpt-", "o4-")):
        return ("openai", key)
    # Groq llama heuristic
    if "llama" in key:
        return ("groq", key)

    # Fallback to Gemini default
    return ("gemini", GEMINI_MODEL_DEFAULT)


# =========================
# EMBEDDINGS + VECTORSTORE
# =========================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def get_vectorstore(host: str, port: int, collection: str):
    try:
        vs = Milvus(
            embedding_function=get_embeddings(),
            collection_name=collection,
            connection_args={"host": host, "port": port},
            vector_field="embedding"  # Specify your correct vector field name
        )
        return vs
    except Exception as e:
        st.error(f"Milvus init error: {e}")
        return None

database: Optional[Milvus] = get_vectorstore(HOST, PORT, COLLECTION_NAME)


# =========================
# SQLITE CHAT STORE
# =========================
class ChatHistoryStore:
    """
    SQLite store for chat history.
    Schema: messages(session_id TEXT, role TEXT('user'|'assistant'), content TEXT, ts REAL)
    """
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
# RAG: RETRIEVAL + PROMPTS
# =========================
def _retrieve_context(query: str, k: int = 5):
    if not database:
        return "", []
    try:
        docs = database.similarity_search(query, k=k)
    except Exception as e:
        st.warning(f"[Milvus] similarity_search error: {e}")
        return "", []
    context_text = "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))
    sources = [
        {
            "page": (d.metadata or {}).get("page"),
            "source": (d.metadata or {}).get("source") or (d.metadata or {}).get("file"),
            "preview": (d.page_content or "")[:300],
        } for d in docs
    ]
    return context_text, sources

def _make_parts(
    query: str,
    context_text: str,
    extracted_text: str,
    uploads: Optional[List[Dict[str, Any]]] = None,
    history_text: str = ""
):
    uploads = uploads or []
    parts: List[Dict[str, Any]] = []

    # Compose the prompt using the provided template
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
# LLM EXECUTION (multi-provider)
# =========================
def _parts_to_text(parts: List[Dict[str, Any]]) -> str:
    """Flatten our Gemini-style parts into a plain text prompt for providers that need it."""
    chunks = []
    for p in parts:
        if "text" in p:
            chunks.append(p["text"])
    return "\n\n".join(chunks)

def _run_gemini(parts: List[Dict[str, Any]], model_name: str) -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(parts)
    return (getattr(resp, "text", "") or "").strip()

def _run_openai(parts: List[Dict[str, Any]], model_name: str) -> str:
    if not _HAS_OPENAI:
        st.error("OpenAI client not installed. Run: pip install openai")
        return "OpenAI client not installed."
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in environment.")
        return "Missing OPENAI_API_KEY."

    client = OpenAI(api_key=OPENAI_API_KEY)
    content_text = _parts_to_text(parts)
    # Simple, text-only message for broad compatibility
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content_text}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def _run_groq(parts: List[Dict[str, Any]], model_name: str) -> str:
    if not _HAS_GROQ:
        st.error("Groq client not installed. Run: pip install groq")
        return "Groq client not installed."
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY in environment.")
        return "Missing GROQ_API_KEY."

    client = Groq(api_key=GROQ_API_KEY)
    content_text = _parts_to_text(parts)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content_text}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def _run_any_model(parts: List[Dict[str, Any]], selected_label_or_id: str) -> str:
    provider, model_id = resolve_model(selected_label_or_id)
    if provider == "gemini":
        return _run_gemini(parts, model_id)
    if provider == "openai":
        return _run_openai(parts, model_id)
    if provider == "groq":
        return _run_groq(parts, model_id)
    # Fallback
    return _run_gemini(parts, GEMINI_MODEL_DEFAULT)


# =========================
# PUBLIC PIPELINE
# =========================
def resultDocuments(
    query: str,
    extracted_text: str = "",
    uploads: Optional[List[Dict[str, Any]]] = None,
    model: str = GEMINI_MODEL_DEFAULT,
    session_id: str = "default",
):
    uploads = uploads or []
    context_text, sources = _retrieve_context(query)

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

    answer = _run_any_model(parts, model)

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

    # Build the model selector
    all_labels = list(MODEL_CATALOG.keys())
    # Ensure the default Gemini is in the list and selected
    default_label = ALIAS_NORMALIZE.get(GEMINI_MODEL_DEFAULT, None)
    default_index = 0
    if default_label and default_label in all_labels:
        default_index = all_labels.index(default_label)
    elif "Gemini 2.5 Flash" in all_labels:
        default_index = all_labels.index("Gemini 2.5 Flash")

    selected_label = st.selectbox("Select Model", all_labels, index=default_index)

    # Provider hints
    prov, mid = resolve_model(selected_label)
    if prov == "openai" and not OPENAI_API_KEY:
        st.warning("OpenAI model selected, but OPENAI_API_KEY is not set.", icon="⚠️")
    if prov == "groq" and not GROQ_API_KEY:
        st.warning("Groq model selected, but GROQ_API_KEY is not set.", icon="⚠️")

    st.caption("Change then click 'Reconnect' to re-init the vector store.")
    if st.button("🔄 Reconnect Milvus"):
        globals()["database"] = get_vectorstore(host, port, collection)
        st.success("Reconnected.")

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

    # Render chat history (scrolls with page)
    history_msgs = CHAT_DB.recent(st.session_state.get("session_id", "default"), limit=MAX_HISTORY_MSGS)
    for m in history_msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

    # Chat input
    user_query = st.chat_input("Ask something (e.g., 'PPE for bridge deck pour near traffic') …")
    if user_query:
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
                model=selected_label,  # pass the label; router resolves it
                session_id=st.session_state.get("session_id", "default"),
            )

        # Show the assistant message
        with st.chat_message("assistant"):
            st.write(resp["text"])

        # Save latest sources for right column display
        st.session_state["last_sources"] = resp.get("sources", [])

with col_docs:
    st.markdown("### 📚 Retrieved Documents")
    st.caption("Top-k chunks retrieved from Milvus for the latest question.")
    sources = st.session_state.get("last_sources", [])
    if not sources:
        st.info("Ask a question to see retrieved documents here.")
    else:
        # Show each retrieved doc in an expander
        for i, s in enumerate(sources, start=1):
            label = f"{i}. {s.get('source') or 'unknown'}"
            with st.expander(label, expanded=(i == 1)):
                st.markdown(f"**Page:** {s.get('page', '—')}")
                preview = s.get("preview", "")
                st.write(preview if preview else "_(no preview)_")
