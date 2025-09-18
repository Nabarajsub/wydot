# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# ---- RAG libs ----
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.milvus import Milvus

# ---- LLM (Gemini) ----
import google.generativeai as genai

# =========================
# ENV + CONFIG
# =========================
load_dotenv()

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "19530"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "wyodotspecs")
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "./chat_history.sqlite3")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# How much recent chat to include each LLM turn
MAX_HISTORY_MSGS = 20
HISTORY_PAIRS_FOR_PROMPT = 6

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
            vector_field="embedding"  # Specify the correct vector field name
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

    # -------- NEW: session list, counts, and paged read --------
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Returns sessions ordered by most recent activity:
        [{"session_id": "...", "count": N, "last_ts": 1712345678.0}, ...]
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT session_id, COUNT(*) AS c, MAX(ts) AS last_ts "
                "FROM messages GROUP BY session_id ORDER BY last_ts DESC"
            )
            rows = cur.fetchall()
        return [{"session_id": r[0], "count": r[1], "last_ts": r[2]} for r in rows]

    def count_session(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
            )
            (cnt,) = cur.fetchone()
        return cnt or 0

    def read_page(self, session_id: str, offset: int = 0, limit: int = 50,
                  search: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return a page of messages in chronological order.
        Optional simple LIKE search across content.
        """
        q = (
            "SELECT role, content, ts FROM messages "
            "WHERE session_id=? {where} ORDER BY id ASC LIMIT ? OFFSET ?"
        )
        params: List[Any] = [session_id]
        where = ""
        if search:
            where = "AND content LIKE ? "
            params.append(f"%{search}%")
        params.extend([limit, offset])
        with self._lock:
            cur = self._conn.execute(q.format(where=where), params)
            rows = cur.fetchall()
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

def _run_gemini(parts: List[Dict[str, Any]], model_name: str = GEMINI_MODEL_DEFAULT) -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(parts)
    return (getattr(resp, "text", "") or "").strip()

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

    answer = _run_gemini(parts, model_name=model)
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
    model_options = [
        ("Gemini 2.5 Flash (Default)", GEMINI_MODEL_DEFAULT),
        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
        ("Gemini 2.5 Flash-Lite", "gemini-2.5-flash-lite"),
    ]
    model_labels = [opt[0] for opt in model_options]
    selected_label = st.selectbox("Select Model", model_labels, index=0)
    model_choice = dict(model_options)[selected_label]

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
    st.markdown("## 🗂️ History Browser")
    # Pull sessions list
    sessions_info = CHAT_DB.list_sessions()
    session_labels = [f"{s['session_id']}  (msgs: {s['count']}, last: {datetime.fromtimestamp(s['last_ts']).strftime('%Y-%m-%d %H:%M:%S')})"
                      for s in sessions_info] or ["— no sessions —"]
    session_keys = [s['session_id'] for s in sessions_info] or ["default"]
    selected_hist_ix = st.selectbox("Select a session to view", range(len(session_keys)), format_func=lambda i: session_labels[i])
    selected_hist_session = session_keys[selected_hist_ix] if session_keys else "default"

    st.session_state["history_view_session"] = selected_hist_session

    st.markdown("---")
    st.markdown("## 📎 Uploads & Extra Text")
    uploaded_files = st.file_uploader(
        "Optional: attach images/audio/video/docs (sent to the model as raw bytes)",
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

    # Render chat history (scrolls with page) for the ACTIVE chat session
    active_sid = st.session_state.get("session_id", "default")
    history_msgs = CHAT_DB.recent(active_sid, limit=MAX_HISTORY_MSGS)
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
                model=model_choice,
                session_id=active_sid,
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

    st.markdown("---")
    st.markdown("### 🗂️ Chat History Browser")
    view_sid = st.session_state.get("history_view_session", "default")

    # Controls
    total_msgs = CHAT_DB.count_session(view_sid)
    st.caption(f"Session **{view_sid}** • total messages: **{total_msgs}**")

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        page_size = st.number_input("Page size", min_value=10, max_value=500, value=50, step=10, key="hist_page_size")
    with col_b:
        search_term = st.text_input("Search (optional)", value="", key="hist_search")
    with col_c:
        # Keep offset in session state per session id
        offset_key = f"hist_offset::{view_sid}"
        if offset_key not in st.session_state:
            st.session_state[offset_key] = 0

        # Paging buttons
    col_p1, col_p2, col_p3 = st.columns([1,1,4])
    with col_p1:
        if st.button("⬅️ Prev"):
            st.session_state[offset_key] = max(0, st.session_state[offset_key] - int(page_size))
    with col_p2:
        if st.button("Next ➡️"):
            next_off = st.session_state[offset_key] + int(page_size)
            if next_off < max(0, total_msgs - 1):
                st.session_state[offset_key] = next_off
    with col_p3:
        st.write("")  # spacer

    offset = st.session_state[offset_key]

    # Load the page (chronological)
    page = CHAT_DB.read_page(view_sid, offset=offset, limit=int(page_size), search=(search_term or None))
    if not page:
        st.info("No messages to display for this slice.")
    else:
        # Render like a chat (chronological)
        for msg in page:
            ts_str = datetime.fromtimestamp(msg["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(f"_{ts_str}_")
                st.write(msg["content"])

        # Download the currently viewed slice
        export_payload = json.dumps(
            {"session_id": view_sid, "offset": offset, "limit": int(page_size),
             "search": search_term or "", "messages": page},
            ensure_ascii=False, indent=2
        ).encode("utf-8")
        st.download_button(
            "⬇️ Download this slice (JSON)",
            data=export_payload,
            file_name=f"chat_history_{view_sid}_{offset}_{int(page_size)}.json",
            mime="application/json"
        )
