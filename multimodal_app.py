#!/usr/bin/env python3
import os, json, tempfile
from typing import Optional, List, Dict, Any
import mimetypes

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from pymilvus import connections, Collection

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ------------ Config ------------
GEN_MODEL    = "gemini-2.0-flash"
DEFAULT_COLL = os.getenv("MILVUS_COLLECTION", "mmrag_chunks_v1")
MILVUS_HOST  = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT  = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI   = os.getenv("MILVUS_URI")
DEFAULT_STORE_DIR = os.getenv("STORE_DIR", "./utils_multimodal/rag_store")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")  # fallback

app = FastAPI(title="WYDOT  employee bot")

# ------------ Helpers ------------
def connect_milvus():
    if MILVUS_URI:
        connections.connect(alias="default", uri=MILVUS_URI,
                            user=os.getenv("MILVUS_USER"), password=os.getenv("MILVUS_PASSWORD"))
    else:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

def read_manifest_embed_model(store_dir: str, doc_hash: Optional[str]) -> str:
    """If a manifest exists, use its embed_model to avoid mismatch."""
    try:
        if doc_hash:
            path = os.path.join(store_dir, f"{doc_hash}.manifest.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f).get("embed_model", DEFAULT_EMBED_MODEL)
        else:
            for name in os.listdir(store_dir):
                if name.endswith(".manifest.json"):
                    with open(os.path.join(store_dir, name), "r", encoding="utf-8") as f:
                        return json.load(f).get("embed_model", DEFAULT_EMBED_MODEL)
    except Exception:
        pass
    return DEFAULT_EMBED_MODEL

def embed_query(q: str, model_name: str) -> np.ndarray:
    embs = client.models.embed_content(model=model_name, contents=[q])
    v = np.array(embs.embeddings[0].values, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)  # normalize for IP≈cosine
    return v

def upload_path_as_file(path: str):
    # Upload by filesystem path (SDK infers mimetype)
    return client.files.upload(file=path)

def upload_user_image(user_img: Optional[UploadFile]):
    """Save upload to a temp file and upload by path (compatible across SDK versions)."""
    if not user_img:
        return None
    ext = os.path.splitext(user_img.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        try:
            user_img.file.seek(0)
        except Exception:
            pass
        tmp.write(user_img.file.read())
        tmp_path = tmp.name
    try:
        return client.files.upload(file=tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def caption_uploaded_file(file_ref) -> str:
    """Generate a concise caption of the uploaded image focused on tables/charts/labels/numbers."""
    prompt = ("Briefly describe this image focusing on any tables or charts: titles, axes labels, "
              "legends, key numbers. ≤60 words.")
    res = client.models.generate_content(
        model=GEN_MODEL,
        contents=[file_ref, prompt],
        config={"temperature": 0.0, "max_output_tokens": 120},
    )
    return (res.text or "").strip()

def build_prompt(context_block: str, question: str) -> str:
    # EXACT system prompt (as you wrote it)
    return (
        "You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).\n"
        "Answer the question from the given context. Ensure clarity, brevity, and human-like responses.\n"
        f"Context inside double backticks:``{context_block}``\n"
        f"Question inside triple backticks:```{question}```\n"
        "If question is out of scope, answer it based on your role.\n"
        "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE."
    )

def milvus_search(query: str, top_k: int, doc_hash: Optional[str], embed_model: str, collection_name: str):
    connect_milvus()
    coll = Collection(collection_name)
    coll.load()

    expr = f'doc_hash == "{doc_hash}"' if doc_hash else None
    qv = embed_query(query, embed_model)

    res = coll.search(
        data=[qv.tolist()],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=top_k,
        expr=expr,
        output_fields=["doc_hash", "page", "kind", "text", "page_image_file"]
    )

    hits = []
    if res and len(res[0]) > 0:
        for hit in res[0]:
            e = hit.entity
            hits.append({
                "score": float(hit.distance),
                "doc_hash": e.get("doc_hash"),
                "page": int(e.get("page")),
                "kind": e.get("kind"),
                "text": e.get("text"),
                "page_image_file": e.get("page_image_file"),
            })
    return hits

def build_context_and_images(hits: List[Dict[str, Any]], max_images: int = 3):
    # Build context text
    ctx_lines = []
    for h in hits:
        tag = "[IMAGE]" if h["kind"] == "image" else "[TEXT]"
        ctx_lines.append(f"({tag} p.{h['page']} | score={h['score']:.3f})\n{h['text'] or ''}")
    context_block = "\n\n---\n".join(ctx_lines) if ctx_lines else "no context retrieved"

    # Deduplicate pages and upload up to N page images
    used_pages, used_files = set(), []
    for h in hits:
        if h["page"] not in used_pages and h.get("page_image_file") and os.path.exists(h["page_image_file"]):
            used_pages.add(h["page"])
            used_files.append(h["page_image_file"])
            if len(used_files) >= max_images:
                break
    uploaded_imgs = [upload_path_as_file(p) for p in used_files]
    return context_block, uploaded_imgs, used_files

# ------------ API models ------------
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    store_dir: str = DEFAULT_STORE_DIR
    doc_hash: Optional[str] = None
    collection: str = DEFAULT_COLL

# ------------ Endpoint (multipart: upload image + ask) ------------
@app.post("/chat_form")
async def chat_form(
    query: str = Form(...),
    top_k: int = Form(5),
    doc_hash: Optional[str] = Form(None),
    store_dir: str = Form(DEFAULT_STORE_DIR),
    collection: str = Form(DEFAULT_COLL),
    user_image: UploadFile | None = File(None),
):
    # Upload user image (optional)
    ui_file = upload_user_image(user_image)

    # If user image present, caption it and append to the query to drive image-aware retrieval
    embed_model = read_manifest_embed_model(store_dir, doc_hash)
    if ui_file:
        img_cap = caption_uploaded_file(ui_file)
        query_for_search = f"{query}\n\nImage clue: {img_cap}"
    else:
        query_for_search = query

    # Retrieve from Milvus
    hits = milvus_search(query_for_search, top_k, doc_hash, embed_model, collection)

    # Build context and gather top page images
    ctx, page_imgs, used_files = build_context_and_images(hits)

    # Build the system prompt; attach retrieved page images and the user image (if any)
    prompt = build_prompt(ctx, query)
    uploads = page_imgs[:]
    if ui_file:
        uploads.append(ui_file)

    # Generate answer grounded in retrieved pages (+ user image attached)
    res = client.models.generate_content(
        model=GEN_MODEL,
        contents=[prompt] + uploads,
        config={"temperature": 0.2, "max_output_tokens": 120},
    )
    answer = (res.text or "").strip()

    return {
        "answer": answer,
        "used_pages": [os.path.basename(p) for p in used_files],
        "user_image": user_image.filename if user_image else None,
        "hits": hits,
        "embed_model": embed_model,
        "augmented_query": query_for_search if ui_file else query,
    }
