#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_vectorstore_zilliz.py

Headless script to create/fill a Milvus (Zilliz Cloud) vector store with Gemini embeddings.

ENV you likely want to set (PowerShell examples):
  $env:MILVUS_URI = "https://in01-f2461fb40555142.gcp-us-west1.vectordb.zillizcloud.com:443"
  $env:MILVUS_TOKEN = "<YOUR_ZILLIZ_TOKEN>"
  $env:MILVUS_COLLECTION = "wydotspec_llamaparse"

  # Input data
  $env:JSONL_PATH = "C:\\path\\to\\Wyoming_2021_WYDOT_cleaned.jsonl"

  # Embeddings
  $env:GEMINI_EMBED_MODEL = "text-embedding-004"   # or "gemini-embedding-001"
  $env:GOOGLE_CLOUD_API_KEY = "<your-google-genai-api-key>"  # or GEMINI_API_KEY / GOOGLE_API_KEY
  # If you prefer Vertex (ADC/service acct):
  # $env:GOOGLE_GENAI_USE_VERTEXAI = "true"
  # $env:GOOGLE_CLOUD_PROJECT = "quiet-era-401008"
  # $env:GOOGLE_CLOUD_LOCATION = "us-central1"

Optional:
  $env:DROP_IF_EXISTS = "false"   # set "true" to drop and recreate collection
  $env:CHUNK_SIZE = "1500"
  $env:CHUNK_OVERLAP = "250"
  $env:MIN_CHARS = "20"
  $env:EMBED_CACHE_PATH = "gemini_embed_cache.sqlite"

Run:
  python build_vectorstore_zilliz.py
"""

import os, re, sys, time, random, sqlite3, hashlib, traceback
from typing import List, Dict, Any, Iterable, Optional

import ujson as json
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# -------- Milvus / Zilliz --------
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)

# -------- Google GenAI (embeddings) --------
from google import genai
from google.genai import types as gtypes
try:
    from google.genai import errors as genai_errors
except Exception:
    genai_errors = None


# ================= Config & ENV =================
load_dotenv()

MILVUS_URI       = os.getenv("MILVUS_URI")
MILVUS_TOKEN     = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME  = os.getenv("MILVUS_COLLECTION", "wydotspec_llamaparse")

JSONL_PATH       = os.getenv("JSONL_PATH")  # required
EMBED_MODEL      = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

USE_VERTEX       = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes", "on"}
PROJECT          = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
LOCATION         = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION") or "us-central1"
API_KEY          = os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

DROP_IF_EXISTS   = os.getenv("DROP_IF_EXISTS", "false").strip().lower() in {"1","true","yes","on"}
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "250"))
MIN_CHARS        = int(os.getenv("MIN_CHARS", "20"))
CACHE_PATH       = os.getenv("EMBED_CACHE_PATH", "gemini_embed_cache.sqlite")

VECTOR_FIELD     = "vector"
METRIC_TYPE      = "COSINE"

# Input JSONL text keys and caps
TEXT_KEYS        = ("text", "content", "page_content", "string")
MAX_SECTION_LEN  = 1024
MAX_SOURCE_LEN   = 1024
MAX_CONTENT_LEN  = 16384


# ================= Helpers: IO & chunking =================
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue

def extract_text(d: Dict[str, Any]) -> str:
    for k in TEXT_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # Try typical LlamaParse-like structure
    if isinstance(d.get("elements"), list):
        parts = []
        for el in d["elements"]:
            t = el.get("text") or el.get("content") or el.get("string")
            if isinstance(t, str) and t.strip():
                parts.append(t)
        if parts:
            return "\n".join(parts)

    # Other fallbacks
    parts = []
    for k in ("header", "footer", "table_text", "figure_text", "paragraphs"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
        elif isinstance(v, list):
            parts.extend([x for x in v if isinstance(x, str)])
    return "\n".join(parts) if parts else ""

def extract_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    # doc_id (stable-ish)
    for k in ("id", "doc_id", "document_id", "file_id", "uuid"):
        if k in d and isinstance(d[k], (str, int)):
            meta["doc_id"] = str(d[k]); break
    # common metadata fields
    if "metadata" in d and isinstance(d["metadata"], dict):
        md = d["metadata"]
        for k in ("source", "file_name", "pdf_name", "path", "url",
                  "division", "section", "subsection", "page"):
            if k in md:
                meta[k] = md[k]
    for k in ("source", "file_name", "pdf_name", "path",
              "division", "section", "subsection"):
        if k in d and k not in meta:
            meta[k] = d[k]
    pg = d.get("page") or (d.get("metadata", {}).get("page") if isinstance(d.get("metadata"), dict) else None)
    if isinstance(pg, (int, float)) and pg >= 0:
        meta["page"] = int(pg)
    if "source" not in meta:
        meta["source"] = d.get("source") or d.get("file_name") or d.get("pdf_name") or "unknown"
    if "doc_id" not in meta:
        meta["doc_id"] = hashlib.sha1((meta["source"] + "|" + str(meta.get("page",""))).encode("utf-8")).hexdigest()
    return meta

def split_into_chunks(text: str, chunk_size: int = 1500, chunk_overlap: int = 250) -> List[str]:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= chunk_size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            nl = chunk.rfind("\n\n")
            if nl > chunk_size * 0.6:
                end = start + nl
                chunk = text[start:end]
        out.append(chunk)
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
    return out


# ================= Google GenAI client & embeddings =================
def make_genai_client() -> genai.Client:
    if USE_VERTEX:
        if not PROJECT:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT when using Vertex AI (ADC/service account).")
        return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    if not API_KEY:
        raise RuntimeError("Set GOOGLE_CLOUD_API_KEY (or GOOGLE_API_KEY/GEMINI_API_KEY) for google-genai.")
    return genai.Client(api_key=API_KEY)

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
    # try JSON conversion via SDK
    try:
        import json as _json
        js = _json.loads(gtypes.to_json(resp))
        return _extract_values_any(js)
    except Exception:
        pass
    raise ValueError("Unexpected embedding response (no 'values').")

def _should_retry(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(k in msg for k in ["unavailable", "overloaded", "timeout", "timed out"]):
        return True
    if "rate" in msg and "limit" in msg:
        return True
    code = getattr(exc, "status_code", None)
    if code in (429, 500, 503):
        return True
    if genai_errors is not None and isinstance(exc, getattr(genai_errors, "ServerError", tuple())):
        try:
            c = getattr(exc, "status_code", None)
            return c in (429, 500, 503)
        except Exception:
            return True
    return False

def backoff_sleep(attempt: int, base: float, cap: float):
    delay = min(cap, base * (2 ** attempt))
    delay *= 0.5 + random.random()  # jitter
    time.sleep(delay)

def embed_one_with_retry(client: genai.Client, text: str, model: str, desired_dim: Optional[int],
                         task_type: str, max_retries: int = 6, base_delay: float = 1.0, max_delay: float = 20.0) -> np.ndarray:
    last_exc = None
    cfg = gtypes.EmbedContentConfig(task_type=task_type)
    if desired_dim:
        cfg.output_dimensionality = int(desired_dim)
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.embed_content(
                model=model,
                contents=text,
                config=cfg,
            )
            vals = _extract_values_any(resp)
            arr = np.asarray(vals, dtype=np.float32)
            # normalize for COSINE if model returns non-unit vectors
            n = float(np.linalg.norm(arr))
            if n > 0:
                arr = arr / n
            return arr
        except Exception as e:
            last_exc = e
            if attempt >= max_retries or not _should_retry(e):
                raise
            backoff_sleep(attempt, base_delay, max_delay)
    raise last_exc  # should not happen


# ================= SQLite cache for embeddings =================
class EmbedCache:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                dim       INTEGER NOT NULL,
                vec       BLOB NOT NULL
            )
        """)
        self.conn.commit()

    @staticmethod
    def make_key(model: str, dim: int, task: str, text_hash: str) -> str:
        return hashlib.sha1(f"{model}|{dim}|{task}|{text_hash}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        cur = self.conn.cursor()
        cur.execute("SELECT dim, vec FROM embeddings WHERE cache_key=?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        dim = int(row[0])
        buf = row[1]
        arr = np.frombuffer(buf, dtype=np.float32)
        if arr.shape[0] != dim:
            return None
        # ensure normalized
        n = float(np.linalg.norm(arr))
        if n > 0:
            arr = arr / n
        return arr

    def put(self, key: str, vec: np.ndarray):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO embeddings (cache_key, dim, vec) VALUES (?, ?, ?)",
            (key, int(vec.shape[0]), memoryview(vec.astype(np.float32).tobytes()))
        )
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# ================= Milvus collection helpers =================
def get_existing_vector_dim(col: Collection) -> Optional[int]:
    try:
        for f in col.schema.fields:
            if f.name == VECTOR_FIELD:
                params = getattr(f, "params", {}) or {}
                if "dim" in params:
                    return int(params["dim"])
                if hasattr(f, "dim"):
                    return int(f.dim)
        return None
    except Exception:
        return None

def ensure_collection(collection_name: str, dim: int, drop_if_exists: bool = False) -> Collection:
    if utility.has_collection(collection_name):
        if drop_if_exists:
            utility.drop_collection(collection_name)
        else:
            col = Collection(collection_name)
            ex_dim = get_existing_vector_dim(col)
            if ex_dim is not None and ex_dim != dim:
                raise RuntimeError(
                    f"Collection '{collection_name}' exists with dim={ex_dim}, but embeddings require dim={dim}. "
                    f"Set DROP_IF_EXISTS=true or change MILVUS_COLLECTION."
                )
            return col

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="hash",     dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id",   dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="page",     dtype=DataType.INT64),
        FieldSchema(name="section",  dtype=DataType.VARCHAR, max_length=MAX_SECTION_LEN),
        FieldSchema(name="source",   dtype=DataType.VARCHAR, max_length=MAX_SOURCE_LEN),
        FieldSchema(name="content",  dtype=DataType.VARCHAR, max_length=MAX_CONTENT_LEN),
        FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="WYDOT chunks (Gemini)")
    col = Collection(name=collection_name, schema=schema, using="default", shards_num=2)
    return col

def create_hnsw_index(col: Collection):
    col.create_index(
        field_name=VECTOR_FIELD,
        index_params={"index_type": "HNSW", "metric_type": METRIC_TYPE, "params": {"M": 16, "efConstruction": 200}},
    )

def insert_batches(col: Collection, rows: List[Dict[str, Any]], batch_size: int = 512) -> None:
    # IMPORTANT: No PK list (auto_id=True)
    for i in range(0, len(rows), batch_size):
        b = rows[i:i+batch_size]
        data = [
            [r["hash"] for r in b],
            [r["doc_id"] for r in b],
            [r["chunk_id"] for r in b],
            [int(r.get("page", -1)) for r in b],
            [r.get("section", "")[:MAX_SECTION_LEN] for r in b],
            [r.get("source", "")[:MAX_SOURCE_LEN] for r in b],
            [r["content"][:MAX_CONTENT_LEN] for r in b],
            [r["vector"] for r in b],
        ]
        col.insert(data)
    col.flush()


# ================= Main pipeline =================
def main():
    if not MILVUS_URI or not MILVUS_TOKEN:
        print("[FATAL] Set MILVUS_URI and MILVUS_TOKEN for Zilliz Cloud.")
        sys.exit(1)
    if not JSONL_PATH or not os.path.exists(JSONL_PATH):
        print(f"[FATAL] JSONL not found: {JSONL_PATH}")
        sys.exit(1)

    print(f"[connect] Milvus/Zilliz: {MILVUS_URI}")
    connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)

    print("[genai] creating client …")
    client = make_genai_client()

    # ---------- PROBE (single call) ----------
    print("[probe] single embedding to detect actual dim …")
    probe_vec = embed_one_with_retry(
        client, "WYDOT specification probe", EMBED_MODEL, desired_dim=None, task_type="RETRIEVAL_DOCUMENT"
    )
    actual_dim = int(probe_vec.shape[0])
    print(f"[probe] detected embedding dim={actual_dim}")

    # ---------- Ensure collection BEFORE bulk embedding ----------
    print(f"[schema] ensure collection '{COLLECTION_NAME}' with dim={actual_dim} …")
    try:
        col = ensure_collection(COLLECTION_NAME, dim=actual_dim, drop_if_exists=DROP_IF_EXISTS)
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        print("Set DROP_IF_EXISTS=true (or change MILVUS_COLLECTION) and re-run. No bulk embedding was done.")
        sys.exit(2)

    # ---------- Plan chunks ----------
    print(f"[read] {JSONL_PATH}")
    seen_hash = set()
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    total_records = 0
    for rec in tqdm(read_jsonl(JSONL_PATH), desc="reading JSONL"):
        total_records += 1
        text = extract_text(rec)
        if not text or len(text.strip()) < MIN_CHARS:
            continue
        meta = extract_meta(rec)
        section = (meta.get("section") or meta.get("subsection") or "")[:MAX_SECTION_LEN]
        source  = (meta.get("source") or "unknown")[:MAX_SOURCE_LEN]

        chunks = split_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for j, ch in enumerate(chunks):
            ch = ch.strip()
            if len(ch) < MIN_CHARS:
                continue
            h = hashlib.sha1((meta["doc_id"] + "|" + ch).encode("utf-8")).hexdigest()
            if h in seen_hash:
                continue
            seen_hash.add(h)
            texts.append(ch)
            metas.append({
                "hash": h,
                "doc_id": meta["doc_id"],
                "chunk_id": f"{meta['doc_id']}::{j:06d}",
                "page": int(meta.get("page", -1)),
                "section": section,
                "source": source,
                "content": ch[:MAX_CONTENT_LEN],
            })

    print(f"[plan] records={total_records}, chunks_for_embedding={len(texts)}, dim={actual_dim}")
    if not texts:
        print("[done] No chunks to embed. Exiting.")
        sys.exit(0)

    # ---------- Cache + embed ----------
    cache = EmbedCache(CACHE_PATH)
    try:
        vecs: List[np.ndarray] = []
        for ch, meta in tqdm(list(zip(texts, metas)), desc="embed+cache", total=len(texts)):
            key = EmbedCache.make_key(EMBED_MODEL, actual_dim, "RETRIEVAL_DOCUMENT", meta["hash"])
            v = cache.get(key)
            if v is None:
                v = embed_one_with_retry(
                    client, ch, EMBED_MODEL, desired_dim=actual_dim, task_type="RETRIEVAL_DOCUMENT"
                )
                cache.put(key, v)
            vecs.append(v)

        # ---------- Insert into Milvus ----------
        print("[insert] bulk insert (index will be created after) …")
        rows = [{**m, "vector": v.tolist()} for m, v in zip(metas, vecs)]
        insert_batches(col, rows, batch_size=512)

        # ---------- Index ----------
        print("[index] creating HNSW index …")
        create_hnsw_index(col)
        col.load()
        print(f"[done] Inserted {len(rows)} chunks into '{COLLECTION_NAME}' (dim={actual_dim}).")

    finally:
        cache.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL] " + str(e))
        traceback.print_exc()
        sys.exit(1)
