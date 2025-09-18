#!/usr/bin/env python3
import os, io, json, hashlib
from dataclasses import dataclass
from typing import List, Dict
import sys
import numpy as np
from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image
from dotenv import load_dotenv

# ===== Gemini (google-generativeai) =====
import google.generativeai as genai

# ===== Milvus =====
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

# ----------------- CONFIG -----------------
load_dotenv()

# Prefer GOOGLE_API_KEY, fallback to GEMINI_API_KEY
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment/.env")
genai.configure(api_key=API_KEY)

TEXT_EMBED_MODEL = "text-embedding-004"     # legacy; consider 'gemini-embedding-001' if available
GEN_MODEL        = "gemini-2.0-flash"

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI  = os.getenv("MILVUS_URI")  # optional
COLLECTION  = "mmrag_chunks_v1"

# >>>>>> IMPORTANT: Point this to your bundled Poppler 'bin' folder <<<<<<
# Example layout you place inside your repo:
#   <repo_root>/third_party/poppler/Library/bin/pdftoppm.exe
POPPLER_PATH = os.getenv("POPPLER_PATH") or r"C:\poppler-24.07.0\Library\bin"

# ----------------- PDF utils -----------------
def _verify_poppler_path(poppler_path: str):
    if not os.path.isdir(poppler_path):
        raise FileNotFoundError(
            f"POPPLER_PATH not found: {poppler_path}\n"
            "Place Poppler binaries in this folder (it must contain pdftoppm(.exe) / pdftocairo(.exe))."
        )
    # On Windows, pdf2image uses pdftoppm.exe
    ppm = os.path.join(poppler_path, "pdftoppm.exe" if os.name == "nt" else "pdftoppm")
    if not os.path.exists(ppm):
        raise FileNotFoundError(
            f"'pdftoppm' not found in POPPLER_PATH: {poppler_path}\n"
            "Ensure the 'bin' directory of Poppler is provided."
        )

def load_pdf_text(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    out = []
    for p in reader.pages:
        try:
            t = (p.extract_text() or "").strip()
        except Exception:
            t = ""
        out.append(t)
    return out

def render_pdf_pages(pdf_path: str, dpi=180) -> List[Image.Image]:
    # Use explicit Poppler path; no system install required
    _verify_poppler_path(POPPLER_PATH)
    return convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)

def chunk_text(raw_text: str, max_words=500) -> List[str]:
    paras = [p.strip() for p in raw_text.split("\n") if p.strip()]
    chunks, cur, count = [], [], 0
    for p in paras:
        w = len(p.split())
        if count + w > max_words and cur:
            chunks.append("\n".join(cur)); cur, count = [], 0
        cur.append(p); count += w
    if cur: chunks.append("\n".join(cur))
    return chunks

def caption_page_image(img: Image.Image) -> str:
    # With google-generativeai, pass image + prompt
    model = genai.GenerativeModel(GEN_MODEL)
    prompt = (
        "Give a compact caption of this page’s visual content: tables, charts, axes labels, legends, figure titles, "
        "and standout numbers. ≤120 words. If no visual content, say 'no visual content'."
    )
    # You can pass PIL Image directly alongside text
    res = model.generate_content([img, prompt])
    return (getattr(res, "text", None) or "").strip()

def embed_texts(texts: List[str]) -> np.ndarray:
    # google-generativeai uses genai.embed_content for embeddings (single content per call)
    vecs = []
    for t in texts:
        r = genai.embed_content(model=TEXT_EMBED_MODEL, content=t or "")
        v = np.array(r["embedding"], dtype="float32")
        v /= (np.linalg.norm(v) + 1e-12)   # normalize for IP≈cosine
        vecs.append(v)
    return np.vstack(vecs).astype("float32")

# ----------------- Milvus setup -----------------
def connect_milvus():
    if MILVUS_URI:
        connections.connect(alias="default", uri=MILVUS_URI,
                            user=os.getenv("MILVUS_USER"), password=os.getenv("MILVUS_PASSWORD"))
    else:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

def ensure_collection(d: int):
    if utility.has_collection(COLLECTION):
        return Collection(COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=16),          # "text" | "image"
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="page_image_file", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=d),
    ]
    schema = CollectionSchema(fields, description="Multimodal RAG chunks (text+image captions)")
    coll = Collection(COLLECTION, schema=schema)
    coll.create_index(
        field_name="vector",
        index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 128}},
    )
    coll.load()
    return coll

def insert_rows(coll: Collection, rows: List[Dict]):
    coll.insert([
        [r["doc_hash"] for r in rows],
        [r["page"] for r in rows],
        [r["kind"] for r in rows],
        [r["text"] for r in rows],
        [r["page_image_file"] for r in rows],
        np.array([r["vector"] for r in rows], dtype="float32"),
    ])
    coll.flush()

# ----------------- Build index -----------------
def build_index(pdf_path: str, out_dir: str = "rag_store"):
    os.makedirs(out_dir, exist_ok=True)
    connect_milvus()

    doc_hash = hashlib.md5(os.path.abspath(pdf_path).encode()).hexdigest()[:10]

    pages_text = load_pdf_text(pdf_path)
    page_imgs  = render_pdf_pages(pdf_path, dpi=180)

    page_img_files = []
    for i, img in enumerate(page_imgs):
        p = os.path.join(out_dir, f"{doc_hash}_page_{i+1}.png")
        img.save(p, "PNG")
        page_img_files.append(p)

    raw_chunks = []
    for i, t in enumerate(pages_text):
        if t:
            for ck in chunk_text(t, max_words=400):
                raw_chunks.append({
                    "doc_hash": doc_hash, "page": i+1, "kind": "text", "text": ck,
                    "page_image_file": page_img_files[i]
                })
    for i, _ in enumerate(page_imgs):
        cap = caption_page_image(page_imgs[i])
        raw_chunks.append({
            "doc_hash": doc_hash, "page": i+1, "kind": "image", "text": cap,
            "page_image_file": page_img_files[i]
        })

    texts = [c["text"] for c in raw_chunks]
    vecs  = embed_texts(texts)
    for c, v in zip(raw_chunks, vecs):
        c["vector"] = v

    coll = ensure_collection(d=vecs.shape[1])
    insert_rows(coll, raw_chunks)

    manifest = {
        "doc_hash": doc_hash,
        "pdf_path": os.path.abspath(pdf_path),
        "page_images": page_img_files,
        "embed_model": TEXT_EMBED_MODEL,
        "collection": COLLECTION,
        "poppler_path": POPPLER_PATH,
    }
    with open(os.path.join(out_dir, f"{doc_hash}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Indexed '{pdf_path}' to Milvus. {len(raw_chunks)} chunks. doc_hash={doc_hash}")

if __name__ == "__main__":
    pdf_path = r"C:\Users\nsubedi1\Desktop\WYDOT project\data\Wyoming 2021 Standard Specifications for Road and Bridge Construction.pdf"
    out_dir = "rag_store"
    build_index(pdf_path, out_dir)
