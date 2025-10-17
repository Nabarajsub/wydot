# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, sqlite3, threading, json, base64
from typing import List, Dict, Any, Optional, Tuple, Generator
from urllib.parse import urlparse, quote

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ---------- Vertex GenAI (new SDK) ----------
from google import genai
from google.genai import types as gtypes
from google.oauth2 import service_account

# ---------- Milvus (direct) ----------
from pymilvus import connections, utility, Collection

# ---------- Optional community components ----------
try:
    from audio_recorder_streamlit import audio_recorder  # returns WAV bytes
except Exception:
    audio_recorder = None


# =========================================================
# ENV & CONSTANTS
# =========================================================
load_dotenv()

# ---- Zilliz/Milvus Serverless (URI + TOKEN) ----
DEFAULT_MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "https://in03-339fce16acb492f.serverless.aws-eu-central-1.cloud.zilliz.com",
)
DEFAULT_MILVUS_TOKEN = os.getenv(
    "MILVUS_TOKEN",
    "7f85f7a13313474b99a9d5225a2c5c590fbd749bb68c07719266c4889d9121c601dc948a8c9052c53dbd84dcfcbbcb6fbb36b17d",
)
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "wydotspec_llamaparse")

# Optional: where your PDFs are hosted (e.g., https://your-bucket/public/specs/)
PDF_BASE_URL = os.getenv("PDF_BASE_URL", "").rstrip("/")

CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "./chat_history.sqlite3")

# ---- Vertex tuned endpoint ----
PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION") or "us-central1"
TUNED_ENDPOINT = os.getenv("WYDOT_TUNED_ENDPOINT")  # projects/.../locations/.../endpoints/...

if not PROJECT:
    raise RuntimeError("Set GOOGLE_CLOUD_PROJECT (or PROJECT_ID) to your GCP project id.")
if not TUNED_ENDPOINT:
    raise RuntimeError("Set WYDOT_TUNED_ENDPOINT to your Vertex endpoint id (projects/.../endpoints/...).")

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

VECTOR_FIELD = "vector"
METRIC_TYPE = "COSINE"

MAX_HISTORY_MSGS = 20
HISTORY_PAIRS_FOR_PROMPT = 6


# =========================================================
# Service Account auth loader (Streamlit secrets-first)
# =========================================================
CLOUD_SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]

def _creds_from_secrets_dict() -> Optional[service_account.Credentials]:
    """
    Load credentials from Streamlit secrets, using a TOML table:
      [gcp_service_account]
      type="service_account"
      project_id="..."
      private_key="-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
      ...
    """
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            return service_account.Credentials.from_service_account_info(info).with_scopes(CLOUD_SCOPE)
    except Exception as e:
        st.warning(f"[Auth] Failed loading st.secrets['gcp_service_account']: {e}")
    return None

def _creds_from_env_file() -> Optional[service_account.Credentials]:
    """Optional fallback: GOOGLE_APPLICATION_CREDENTIALS file path (not committed)."""
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if key_path and os.path.exists(key_path):
            return service_account.Credentials.from_service_account_file(key_path).with_scopes(CLOUD_SCOPE)
    except Exception as e:
        st.warning(f"[Auth] Failed loading GOOGLE_APPLICATION_CREDENTIALS file: {e}")
    return None

def _creds_from_env_json() -> Optional[service_account.Credentials]:
    """Optional fallback: GCP_SERVICE_ACCOUNT_JSON (raw JSON) or GCP_SERVICE_ACCOUNT_BASE64."""
    try:
        raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
        if raw:
            info = json.loads(raw)
            return service_account.Credentials.from_service_account_info(info).with_scopes(CLOUD_SCOPE)
        b64 = os.getenv("GCP_SERVICE_ACCOUNT_BASE64")
        if b64:
            info = json.loads(base64.b64decode(b64).decode("utf-8"))
            return service_account.Credentials.from_service_account_info(info).with_scopes(CLOUD_SCOPE)
    except Exception as e:
        st.warning(f"[Auth] Failed loading env JSON creds: {e}")
    return None

def load_gcp_credentials() -> Tuple[Optional[service_account.Credentials], str]:
    """
    Order:
      1) st.secrets['gcp_service_account']  (recommended)
      2) GOOGLE_APPLICATION_CREDENTIALS file
      3) GCP_SERVICE_ACCOUNT_JSON / GCP_SERVICE_ACCOUNT_BASE64
      else ADC
    """
    for loader, label in [
        (_creds_from_secrets_dict, "st.secrets:gcp_service_account"),
        (_creds_from_env_file, "GOOGLE_APPLICATION_CREDENTIALS file"),
        (_creds_from_env_json, "env:GCP_SERVICE_ACCOUNT_JSON/BASE64"),
    ]:
        creds = loader()
        if creds:
            return creds, label
    return None, "ADC (no SA key found)"


# ===== Prompt =====
def format_prompt(context_text: str, extracted_text: str, question: str, history_text: str, has_media: bool = False) -> str:
    """
    Media-aware prompt. If no text question but media exists, ask model to infer intent from media.
    """
    base = (
        "You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).\n"
        "Answer the question using the provided context and (if relevant) the prior conversation.\n\n"
        "Conversation so far (most recent turns first):\n"
        "{history}\n\n"
        "Context inside double backticks:``{context}``\n"
    )
    if (question or "").strip():
        tail = (
            "Question inside triple backticks:```{question}```\n"
            "If the question is out of scope, answer based on your role.\n"
            "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
        )
    else:
        inferred = (
            "No explicit text question was provided. Attached media may include audio, video, images, or PDFs.\n"
            "Do the following in order:\n"
            "1) If audio/video is present: transcribe it (brief timestamps only if helpful) and extract the user's intent.\n"
            "2) If images/PDFs are present: describe/parse them (OCR key fields if relevant).\n"
            "3) Use any retrieved context and the transcription/parse to answer succinctly.\n"
            "4) If it's not a question, provide a concise WYDOT-relevant summary or actionable next steps.\n"
            "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
        )
        tail = inferred if has_media else (
            "No explicit question was provided. Provide a brief helpful response based on the context above.\n"
            "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
        )

    return base.format(
        history=(history_text or "(no previous turns)"),
        context=context_text if context_text else extracted_text,
    ) + tail.format(question=question or "")


# =========================================================
# google-genai Client (Vertex endpoint; uses SA credentials)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_genai_client() -> genai.Client:
    creds, label = load_gcp_credentials()
    st.session_state["auth_label"] = label
    if creds:
        return genai.Client(vertexai=True, project=PROJECT, location=LOCATION, credentials=creds)
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)  # ADC


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
# Milvus connection (URI+TOKEN) & schema probe
# =========================================================
@st.cache_resource(show_spinner=False)
def get_milvus_collection(uri: str, token: str, collection: str) -> Tuple[Optional[Collection], Optional[int], Optional[list]]:
    try:
        connections.connect(alias="default", uri=uri, token=token)
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
    import math
    n = math.sqrt(sum(v*v for v in vals))
    if n > 0:
        vals = [v / n for v in vals]
    return vals


# =========================================================
# Retrieval (Milvus direct)
# =========================================================
def milvus_similarity_search(query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    if not query or not query.strip():
        return "", []

    uri = st.session_state.get("milvus_uri", DEFAULT_MILVUS_URI)
    token = st.session_state.get("milvus_token", DEFAULT_MILVUS_TOKEN)
    collection = st.session_state.get("collection", DEFAULT_COLLECTION)

    col, dim, _ = get_milvus_collection(uri, token, collection)
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
                "doc_id": md.get("doc_id"),
                "page": md.get("page"),
                "source": md.get("source"),
                "preview": content[:300] if content else ""
            })
    return "\n\n".join(chunks), sources


# =========================================================
# Build google-genai request (streaming)
# =========================================================
def build_contents_and_config(
    query: str,
    context_text: str,
    extracted_text: str,
    uploads: Optional[List[Dict[str, Any]]],
    history_text: str,
) -> Tuple[list, gtypes.GenerateContentConfig]:
    uploads = uploads or []
    has_media = len(uploads) > 0

    prompt_text = format_prompt(
        context_text=context_text,
        extracted_text=extracted_text,
        question=query or "",
        history_text=history_text,
        has_media=has_media,
    )

    parts = [gtypes.Part.from_text(text=prompt_text)]
    for item in uploads:
        b = item.get("bytes")
        if not b:
            continue
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
    context_text, sources = milvus_similarity_search(query if (query and query.strip()) else "", k=5)
    history_text = get_history_text(session_id, max_pairs=HISTORY_PAIRS_FOR_PROMPT)

    if (query or "").strip():
        add_to_history(session_id, "user", query)
    elif uploads:
        kinds = []
        for u in uploads:
            mt = (u.get("mime") or "").lower()
            if mt.startswith("audio/"): kinds.append("audio")
            elif mt.startswith("video/"): kinds.append("video")
            elif mt.startswith("image/"): kinds.append("image")
            elif mt.endswith("pdf"):     kinds.append("pdf")
            else:                        kinds.append("file")
        add_to_history(session_id, "user", f"(attachments only: {', '.join(kinds)})")

    contents, config = build_contents_and_config(
        query=query or "",
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
# PDF deep-link helpers
# =========================================================
def _looks_like_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False


def build_pdf_url(source: Optional[str], page: Optional[int]) -> Optional[str]:
    if not source:
        return None
    anchor = ""
    if isinstance(page, int) and page >= 1:
        anchor = f"#page={page}"
    if _looks_like_url(source):
        return f"{source}{anchor}"
    if PDF_BASE_URL:
        filename = os.path.basename(str(source))
        return f"{PDF_BASE_URL}/{quote(filename)}{anchor}"
    return None


# =========================================================
# Chat composer (attachments + camera + audio/video record)
# =========================================================
def render_chat_composer() -> Tuple[Optional[str], List[Dict[str, Any]]]:
    st.markdown("""
    <style>
      .composer {padding:10px 12px; border:1px solid #e6e6e6; border-radius:28px; background:#fff;}
      .composer .stTextInput>div>div>input {border:0 !important; outline:none !important; background:transparent !important;}
      .composer .stTextInput>label {display:none;}
      .composer .stButton>button {border-radius:24px; height:40px;}
      .icon-btn button {padding:0 10px; height:36px;}
      .chip {display:inline-block; padding:2px 8px; border-radius:12px; background:#f1f1f5; margin-right:6px; font-size:12px;}
      .right-sticky { position: sticky; top: 70px; max-height: calc(100vh - 90px); overflow-y: auto; }
      .left-scroll  { max-height: calc(100vh - 90px); overflow-y: auto; }
    </style>
    """, unsafe_allow_html=True)

    if "chat_attached_preview" not in st.session_state:
        st.session_state["chat_attached_preview"] = []
    if "camera_image" not in st.session_state:
        st.session_state["camera_image"] = None
    if "audio_bytes" not in st.session_state:
        st.session_state["audio_bytes"] = None
    if "video_file" not in st.session_state:
        st.session_state["video_file"] = None

    with st.form("chat_composer_form", clear_on_submit=True):
        c_plus, c_input, c_mic, c_send = st.columns([0.12, 1, 0.12, 0.15])

        with c_plus:
            with st.popover("➕", use_container_width=False):
                tab_up, tab_cam, tab_aud, tab_vid = st.tabs(["Upload", "Camera", "Record audio", "Record video"])

                with tab_up:
                    attachments = st.file_uploader(
                        "Attach files",
                        type=[
                            "png","jpg","jpeg","webp","gif",
                            "pdf",
                            "mp4","mov","mkv","webm","avi"
                        ],
                        accept_multiple_files=True,
                        help="Add images, PDFs, or videos"
                    )
                    st.caption("Tip: On mobile/Chrome you can pick 'Use camera' to record a video directly.")
                    if attachments is not None:
                        st.session_state["chat_attached_preview"] = attachments

                with tab_cam:
                    cam_img = st.camera_input("Click to take a photo")
                    if cam_img is not None:
                        st.session_state["camera_image"] = cam_img

                with tab_aud:
                    if audio_recorder is not None:
                        st.caption("Press to record / stop:")
                        wav_bytes = audio_recorder(text="", recording_color="#e74c3c", neutral_color="#2ecc71", icon_size="2x")
                        if wav_bytes:
                            st.audio(wav_bytes, format="audio/wav")
                            st.session_state["audio_bytes"] = wav_bytes
                    else:
                        st.info("Install `audio-recorder-streamlit` to record in-browser, or upload a file below.")
                        aud_up = st.file_uploader("Upload audio", type=["wav","mp3","m4a","webm"], accept_multiple_files=False, key="aud_up")
                        if aud_up is not None:
                            st.session_state["audio_bytes"] = aud_up.getvalue()

                with tab_vid:
                    vid_up = st.file_uploader("Record/Upload video", type=["mp4","mov","mkv","webm","avi"], accept_multiple_files=False, key="vid_up")
                    if vid_up is not None:
                        st.session_state["video_file"] = vid_up

            cnt = len(st.session_state.get("chat_attached_preview", []))
            cnt += 1 if st.session_state.get("camera_image") else 0
            cnt += 1 if st.session_state.get("audio_bytes") else 0
            cnt += 1 if st.session_state.get("video_file") else 0
            if cnt:
                st.markdown(f'<span class="chip">{cnt} attachment{"s" if cnt>1 else ""}</span>', unsafe_allow_html=True)

        with c_input:
            message = st.text_input("Ask anything", placeholder="Ask anything related to WYDOT", label_visibility="collapsed", key="composer_text")

        with c_mic:
            with st.popover("🎤", use_container_width=True):
                extra_aud = st.file_uploader("Upload audio (optional)", type=["wav","mp3","m4a","webm"], accept_multiple_files=False, key="extra_aud_up")
                if extra_aud is not None:
                    st.session_state["audio_bytes"] = extra_aud.getvalue()

        with c_send:
            submitted = st.form_submit_button("➤", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            payload: List[Dict[str, Any]] = []

            for f in list(st.session_state.get("chat_attached_preview", [])):
                try:
                    payload.append({
                        "bytes": f.getvalue(),
                        "mime": f.type or "application/octet-stream",
                        "name": f.name,
                    })
                except Exception as e:
                    st.warning(f"Failed to read {getattr(f, 'name', 'file')}: {e}")

            if st.session_state.get("camera_image") is not None:
                cam_file = st.session_state["camera_image"]
                try:
                    payload.append({
                        "bytes": cam_file.getvalue(),
                        "mime": "image/jpeg",
                        "name": "camera.jpg",
                    })
                except Exception as e:
                    st.warning(f"Camera image read failed: {e}")

            if st.session_state.get("audio_bytes"):
                payload.append({
                    "bytes": st.session_state["audio_bytes"],
                    "mime": "audio/wav",
                    "name": "recording.wav",
                })

            if st.session_state.get("video_file") is not None:
                v = st.session_state["video_file"]
                try:
                    payload.append({
                        "bytes": v.getvalue(),
                        "mime": v.type or "video/mp4",
                        "name": v.name or "recording.mp4",
                    })
                except Exception as e:
                    st.warning(f"Video read failed: {e}")

            st.session_state["chat_attached_preview"] = []
            st.session_state["camera_image"] = None
            st.session_state["audio_bytes"] = None
            st.session_state["video_file"] = None

            return (message.strip() if message else None), payload

    return None, []


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="WYDOT employee bot", page_icon="🛣️", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("Wydot employee bot 🛣️")
    st.session_state.setdefault("milvus_uri", DEFAULT_MILVUS_URI)
    st.session_state.setdefault("milvus_token", DEFAULT_MILVUS_TOKEN)
    st.session_state.setdefault("collection", DEFAULT_COLLECTION)

    uri = st.text_input("Milvus URI", st.session_state["milvus_uri"])
    token = st.text_input("Milvus Token", st.session_state["milvus_token"], type="password")
    collection = st.text_input("Milvus collection", st.session_state["collection"])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reconnect Milvus"):
            st.session_state["milvus_uri"] = uri
            st.session_state["milvus_token"] = token
            st.session_state["collection"] = collection
            get_milvus_collection.clear()
            st.success("Reconnected.")
    with c2:
        if st.button("🧹 Clear caches"):
            get_milvus_collection.clear()
            get_chat_store.clear()
            get_genai_client.clear()
            st.toast("Cleared Streamlit caches.", icon="🧹")

    st.markdown("---")
    st.markdown("## 🤖 Vertex endpoint")
    st.code(TUNED_ENDPOINT, language="text")
    st.caption(f"Auth: Service Account ({st.session_state.get('auth_label','unknown')}). Project: {PROJECT}, Location: {LOCATION}")

    if PDF_BASE_URL:
        st.markdown("---")
        st.markdown("## 📄 PDF base")
        st.caption("Using PDF_BASE_URL to build links:")
        st.code(PDF_BASE_URL, language="text")

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
    st.markdown("## 📝 Extra grounding text (optional)")
    extracted_text = st.text_area(
        "Paste OCR/STT/parsed text if you want it included in the context",
        height=140,
        placeholder="Paste text extracted from a PDF, image, or audio transcript…"
    )

# Layout: scrollable left chat, sticky right docs
col_chat, col_docs = st.columns([2, 0.8])

with col_chat:
    st.markdown("<div class='left-scroll'>", unsafe_allow_html=True)
    st.markdown("### 🛣️ WYDOT employee bot")
    st.markdown('<div id="chat_top"></div>', unsafe_allow_html=True)

    history_msgs = CHAT_DB.recent(st.session_state.get("session_id", "default"), limit=MAX_HISTORY_MSGS)
    for m in history_msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

    user_query, uploads_payload = render_chat_composer()

    if user_query or uploads_payload:
        with st.chat_message("user"):
            if user_query:
                st.write(user_query)
            if uploads_payload:
                st.caption("Attachments added.")

        if user_query and user_query.strip():
            _, live_sources = milvus_similarity_search(user_query, k=5)
        else:
            live_sources = []
        st.session_state["last_sources"] = live_sources

        with st.chat_message("assistant"):
            placeholder = st.empty()
            current = ""
            for partial_text, sources in resultDocuments_streaming(
                query=user_query or "",
                extracted_text=extracted_text,
                uploads=uploads_payload,
                session_id=st.session_state.get("session_id", "default"),
            ):
                current = partial_text
                placeholder.markdown(current)

        st.markdown('<div id="chat_bottom"></div>', unsafe_allow_html=True)
        components.html(
            """<script>
            const el = parent.document.getElementById('chat_bottom');
            if (el) el.scrollIntoView({behavior:'smooth', block:'end'});
            </script>""",
            height=0,
        )

    st.markdown('<div id="chat_bottom"></div>', unsafe_allow_html=True)
    components.html(
        """<script>
        const el = parent.document.getElementById('chat_bottom');
        if (el) el.scrollIntoView({behavior:'smooth', block:'end'});
        </script>""",
        height=0,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_docs:
    st.markdown("<div class='right-sticky'>", unsafe_allow_html=True)
    st.markdown("### 📚 Retrieved Documents")
    st.caption("Top-k chunks retrieved for the latest question.")
    sources = st.session_state.get("last_sources", [])

    preview_url = st.session_state.get("pdf_preview_url")
    if preview_url:
        st.markdown("#### 🔎 Inline PDF preview")
        components.iframe(preview_url, height=600)
        st.markdown("---")

    if not sources:
        st.info("Ask a question to see retrieved documents here.")
    else:
        for i, s in enumerate(sources, start=1):
            page = s.get("page")
            source = s.get("source")
            url = build_pdf_url(source, page)

            header = f"{i}. {source or 'unknown'}"
            with st.expander(header, expanded=(i == 1)):
                st.markdown(f"**Page:** {page if isinstance(page, int) and page >= 0 else '—'}")

                c1, c2 = st.columns(2)
                with c1:
                    if url:
                        st.link_button(
                            f"🔗 Open at page {page}" if isinstance(page, int) and page >= 1 else "🔗 Open source",
                            url,
                            help="Opens in a new browser tab",
                            use_container_width=True,
                        )
                    else:
                        st.caption("_No PDF link available (set PDF_BASE_URL or ensure source is a URL)._")
                with c2:
                    if url:
                        if st.button("🖼️ Inline preview", key=f"preview_{i}", use_container_width=True):
                            st.session_state["pdf_preview_url"] = url
                            st.experimental_rerun()
                    else:
                        st.caption(" ")

                preview = s.get("preview", "")
                if url and preview:
                    st.markdown(f"[{preview}]({url})")
                else:
                    st.write(preview or "_(no preview)_")
    st.markdown("</div>", unsafe_allow_html=True)
