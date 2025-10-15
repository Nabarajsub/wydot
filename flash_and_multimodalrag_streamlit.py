# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, sqlite3, threading
from typing import List, Dict, Any, Optional, Tuple, Generator

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ---------- Vertex GenAI (new SDK) ----------
from google import genai
from google.genai import types as gtypes

# ---------- Milvus (direct) ----------
from pymilvus import connections, utility, Collection

# =========================================================
# ENV & CONSTANTS
# =========================================================
load_dotenv()

DEFAULT_HOST = os.getenv("HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("PORT", "19530"))
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "wydotspec_llamaparse")

CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "./chat_history.sqlite3")

PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION") or "us-central1"

TUNED_ENDPOINT = os.getenv("WYDOT_TUNED_ENDPOINT")  # projects/.../locations/.../endpoints/...
if not PROJECT:
    raise RuntimeError("Set GOOGLE_CLOUD_PROJECT to your GCP project id.")
if not TUNED_ENDPOINT:
    raise RuntimeError("Set WYDOT_TUNED_ENDPOINT to your Vertex endpoint id.")

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
VECTOR_FIELD = "vector"
METRIC_TYPE = "COSINE"

MAX_HISTORY_MSGS = 20
HISTORY_PAIRS_FOR_PROMPT = 6

# ===== Your ORIGINAL PROMPT (verbatim) =====
def format_prompt(context_text: str, extracted_text: str, question: str) -> str:
    return (
        "You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).\n"
        "Answer the question from the given context. Ensure clarity, brevity, and human-like responses.\n"
        "Context inside double backticks:``{context}``\n"
        "Question inside triple backticks:```{question}```\n"
        "If question is out of scope, answer it based on your role.\n"
        "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
    ).format(context=context_text if context_text else extracted_text, question=question)

# =========================================================
# google-genai Client (Vertex endpoint; ADC/OAuth auth)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_genai_client() -> genai.Client:
    # Vertex via ADC (service account or `gcloud auth application-default login`)
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

# =========================================================
# Chat history (SQLite)
# =========================================================
class ChatHistoryStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sid_id ON messages (session_id, id)")
        self._conn.commit()

    def add(self, session_id: str, role: str, content: str, ts: Optional[float] = None):
        if not session_id: session_id = "default"
        if ts is None: ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, ts)
            )
            self._conn.commit()

    def recent(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not session_id: session_id = "default"
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

# =========================================================
# Milvus connection & schema probe
# =========================================================
@st.cache_resource(show_spinner=False)
def get_milvus_collection(host: str, port: int, collection: str) -> Tuple[Optional[Collection], Optional[int], Optional[list]]:
    try:
        connections.connect(alias="default", host=host, port=str(port))
    except Exception as e:
        st.error(f"Milvus connect error: {e}")
        return None, None, None

    if not utility.has_collection(collection):
        st.error(f"Milvus collection not found: {collection}")
        return None, None, None

    col = Collection(collection)
    try:
        col.load()
    except Exception as e:
        st.warning(f"Milvus load() warning: {e}")

    dim = None
    fields_summary = []
    for f in col.schema.fields:
        params = getattr(f, "params", {}) or {}
        fields_summary.append({
            "name": f.name,
            "dtype": str(getattr(f, "dtype", "")),
            "is_primary": bool(getattr(f, "is_primary", False)),
            "auto_id": bool(getattr(f, "auto_id", False)),
            "params": dict(params),
        })
        if f.name == VECTOR_FIELD:
            dim = params.get("dim") or getattr(f, "dim", None)
            if dim is not None:
                dim = int(dim)

    if dim is None:
        st.error(f"Could not detect vector dimension for field '{VECTOR_FIELD}'.")
        return None, None, fields_summary

    return col, dim, fields_summary

# =========================================================
# Embeddings via google-genai (Vertex)
# =========================================================
def _extract_values_any(resp) -> List[float]:
    if hasattr(resp, "embedding") and getattr(resp.embedding, "values", None):
        return [float(x) for x in resp.embedding.values]
    if hasattr(resp, "embeddings") and getattr(resp, "embeddings", None):
        e0 = resp.embeddings[0]
        vals = getattr(e0, "values", None)
        if vals is not None:
            return [float(x) for x in vals]
    if isinstance(resp, dict):
        if "embedding" in resp and isinstance(resp["embedding"], dict) and "values" in resp["embedding"]:
            return [float(x) for x in resp["embedding"]["values"]]
        if "embeddings" in resp and isinstance(resp["embeddings"], list) and resp["embeddings"]:
            e0 = resp["embeddings"][0]
            if "values" in e0:
                return [float(x) for x in e0["values"]]
    try:
        import json as _json
        js = _json.loads(gtypes.to_json(resp))
        return _extract_values_any(js)
    except Exception:
        pass
    raise ValueError("Unexpected embedding response structure (no 'values' found).")

@st.cache_resource(show_spinner=False)
def get_embed_client() -> genai.Client:
    return get_genai_client()

def embed_query_vector(text: str, dim: int) -> List[float]:
    client = get_embed_client()
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=gtypes.EmbedContentConfig(
            output_dimensionality=dim,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    vals = _extract_values_any(resp)
    # L2-normalize for cosine
    import math
    n = math.sqrt(sum(v*v for v in vals))
    if n > 0:
        vals = [v / n for v in vals]
    return vals

# =========================================================
# Retrieval (Milvus direct)
# =========================================================
def milvus_similarity_search(query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    host = st.session_state.get("host", DEFAULT_HOST)
    port = st.session_state.get("port", DEFAULT_PORT)
    collection = st.session_state.get("collection", DEFAULT_COLLECTION)

    col, dim, _ = get_milvus_collection(host, port, collection)
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

    chunks, sources = [], []
    if res and len(res) > 0:
        for hit in res[0]:
            md = hit.entity
            content = md.get("content") or ""
            chunks.append(content)
            sources.append({
                "page": md.get("page"),
                "source": md.get("source"),
                "preview": content[:300] if content else ""
            })
    return "\n\n".join(chunks), sources

# =========================================================
# Build google-genai request (streaming) with ORIGINAL PROMPT
# =========================================================
def build_contents_and_config(query: str, context_text: str, extracted_text: str,
                              uploads: Optional[List[Dict[str, Any]]],
                              history_text: str) -> Tuple[list, gtypes.GenerateContentConfig]:
    uploads = uploads or []
    prompt_text = format_prompt(context_text, extracted_text, query)

    parts = [gtypes.Part.from_text(text=prompt_text)]
    # Attach uploads as bytes (if any)
    for item in uploads:
        b = item.get("bytes")
        if not b: continue
        mime = item.get("mime") or "application/octet-stream"
        parts.append(gtypes.Part.from_bytes(data=b, mime_type=mime))

    contents = [gtypes.Content(role="user", parts=parts)]

    config = gtypes.GenerateContentConfig(
        temperature=1,
        top_p=1,
        max_output_tokens=65535,
        safety_settings=[
            gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            gtypes.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            gtypes.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            gtypes.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        # Include thinking_config only if enabled on your endpoint
        thinking_config=gtypes.ThinkingConfig(thinking_budget=-1),
        # No system_instruction since you're using the original prompt block
    )
    return contents, config

# =========================================================
# Pipeline (streaming)
# =========================================================
def resultDocuments_streaming(
    query: str,
    extracted_text: str = "",
    uploads: Optional[List[Dict[str, Any]]] = None,
    session_id: str = "default",
) -> Generator[Tuple[str, List[Dict[str, Any]]], None, None]:
    context_text, sources = milvus_similarity_search(query, k=5)
    history_text = get_history_text(session_id, max_pairs=HISTORY_PAIRS_FOR_PROMPT)
    if query:
        add_to_history(session_id, "user", query)

    contents, config = build_contents_and_config(
        query=query,
        context_text=context_text,
        extracted_text=extracted_text,
        uploads=uploads,
        history_text=history_text,
    )

    client = get_genai_client()
    acc: List[str] = []
    try:
        for chunk in client.models.generate_content_stream(
            model=TUNED_ENDPOINT,
            contents=contents,
            config=config,
        ):
            if getattr(chunk, "text", None):
                acc.append(chunk.text)
                yield "".join(acc), sources
        final_text = "".join(acc).strip()
        if final_text:
            add_to_history(session_id, "assistant", final_text)
    except Exception as e:
        err = f"[Model error] {e}"
        add_to_history(session_id, "assistant", err)
        yield err, sources

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="WYDOT employee bot", page_icon="🛣️", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.setdefault("host", DEFAULT_HOST)
    st.session_state.setdefault("port", DEFAULT_PORT)
    st.session_state.setdefault("collection", DEFAULT_COLLECTION)

    host = st.text_input("Milvus host", st.session_state["host"])
    port = st.number_input("Milvus port", min_value=1, max_value=65535, value=int(st.session_state["port"]), step=1)
    collection = st.text_input("Milvus collection", st.session_state["collection"])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reconnect Milvus"):
            st.session_state["host"] = host
            st.session_state["port"] = int(port)
            st.session_state["collection"] = collection
            get_milvus_collection.clear()
            st.success("Reconnected.")
    with c2:
        if st.button("🧹 Clear caches"):
            get_milvus_collection.clear()
            get_chat_store.clear()
            # also clear genai client caches if needed
            get_genai_client.clear()
            st.toast("Cleared Streamlit caches.", icon="🧹")

    st.markdown("---")
    st.markdown("## 🤖 Vertex endpoint (fixed)")
    st.code(TUNED_ENDPOINT, language="text")
    st.caption("Auth: google-genai via ADC (service account or gcloud user)")

    st.markdown("---")
    st.markdown("## 💬 Session")
    default_sid = st.session_state.get("session_id", "default")
    session_id = st.text_input("Session ID", value=default_sid, help="Use a stable ID per user/thread.")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Set session"):
            st.session_state["session_id"] = session_id
            st.success(f"Session set: {session_id}")
    with d2:
        if st.button("Clear session history"):
            CHAT_DB.clear_session(session_id)
            st.toast("Session history cleared.", icon="🧹")

    st.markdown("---")
    st.markdown("## 📎 Uploads & Extra Text")
    uploaded_files = st.file_uploader(
        "Optional: attach images/audio/video/docs (sent as raw bytes)",
        accept_multiple_files=True
    )
    extracted_text = st.text_area(
        "Optional: paste OCR/STT/parsed text",
        height=140,
        placeholder="Paste text extracted from a PDF, image, or audio transcript…"
    )

col_chat, col_docs = st.columns([2, 1])

with col_chat:
    st.markdown("### 🛣️ WYDOT employee bot (streaming)")
    st.markdown('<div id="chat_top"></div>', unsafe_allow_html=True)

    # Show history
    history_msgs = CHAT_DB.recent(st.session_state.get("session_id", "default"), limit=MAX_HISTORY_MSGS)
    for m in history_msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

    # Input
    user_query = st.chat_input("Ask something (e.g., 'PPE for bridge deck pour near traffic') …")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        # Prepare uploads payload
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

        # Streaming answer
        with st.chat_message("assistant"):
            placeholder = st.empty()
            current = ""
            for partial_text, sources in resultDocuments_streaming(
                query=user_query,
                extracted_text=extracted_text,
                uploads=uploads_payload,
                session_id=st.session_state.get("session_id", "default"),
            ):
                current = partial_text
                placeholder.markdown(current)

        # Save latest sources for right column
        st.session_state["last_sources"] = sources if 'sources' in locals() else []

        # Scroll to bottom
        st.markdown('<div id="chat_bottom"></div>', unsafe_allow_html=True)
        components.html(
            """<script>
            const el = parent.document.getElementById('chat_bottom');
            if (el) el.scrollIntoView({behavior:'smooth', block:'end'});
            </script>""",
            height=0,
        )

    # Keep bottom anchor for reruns
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
                st.write(s.get("preview") or "_(no preview)_")
