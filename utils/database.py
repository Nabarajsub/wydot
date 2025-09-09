# ragspecs_build_milvus.py
import os
import re
from uuid import uuid4
from typing import List

from dotenv import load_dotenv

# --- LangChain imports (handle both old/new namespaces) ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Milvus
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fallback for older LangChain installs
    from langchain.document_loaders import PyPDFLoader
    from langchain.vectorstores.milvus import Milvus
    from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional (hybrid & reranking). Comment out if not needed / not installed.
try:
    from langchain.retrievers import EnsembleRetriever
    HAS_ENSEMBLE = True
except Exception:
    HAS_ENSEMBLE = False

# If you want keyword BM25 (hybrid) and you have langchain_community retrievers:
try:
    from langchain_community.retrievers import BM25Retriever
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# Optional cross-encoder reranking (requires sentence-transformers)
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS = True
except Exception:
    HAS_CROSS = False


# -------------------------------
# 1) Environment & config
# -------------------------------
load_dotenv()
HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "wyospecsnew")
PDF_PATH = os.getenv(
    "PDF_PATH",
    "/Users/uw-user/Desktop/WYDOT/Data/Engineering and technical program/WYDOT Standard Specifications – 2021 Edition (Road & Bridge Construction)/Wyoming 2021 Standard Specifications for Road and Bridge Construction.pdf",
)

# Embedding model (384-dim). Normalize for IP/Cosine search.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Text-splitting parameters (tune for your corpus)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))       # ~600–900 chars often works well
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120")) # ~10–20% overlap

TOP_K = int(os.getenv("TOP_K", "8"))  # retrieval fan-out


# -------------------------------
# 2) Lightweight cleaners/parsers
# -------------------------------
WHITESPACE_RE = re.compile(r"\s+")
SECTION_GUESS_RE = re.compile(
    r"^\s*((Section|SECTION)\s+\d+[\.\-]?\d*|[A-Z][A-Z0-9 \-]{6,})"
)

def clean_text(text: str) -> str:
    """Normalize whitespace, fix hyphenation/linebreak artifacts common in PDFs."""
    # join broken hyphenations like "construc-\ntion" -> "construction"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # replace newlines with spaces, collapse whitespace
    text = WHITESPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    return text

def guess_section_title(chunk_text: str) -> str:
    """Heuristic: take first line/phrase that looks like a section heading or ALL CAPS title."""
    m = SECTION_GUESS_RE.search(chunk_text[:200])  # only look at beginning
    return m.group(0).strip() if m else ""


# -------------------------------
# 3) Build embeddings
# -------------------------------
def build_embeddings():
    # Normalize embeddings -> better cosine/IP behavior in Milvus
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},                 # set "cuda" if available
        encode_kwargs={"normalize_embeddings": True},
    )


# -------------------------------
# 4) Load, clean, split, enrich
# -------------------------------
def load_and_prepare_documents(pdf_path: str) -> List:
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()  # one Document per page

    # Clean page content
    for d in raw_docs:
        d.page_content = clean_text(d.page_content)
        # Make sure some base metadata is present
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source", os.path.basename(pdf_path))
        # PyPDFLoader typically adds "page" index; ensure int
        if "page" in d.metadata:
            try:
                d.metadata["page"] = int(d.metadata["page"])
            except Exception:
                pass

    # Split with overlap at natural boundaries where possible
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # try to cut at paragraph/sentence first
    )
    split_docs = splitter.split_documents(raw_docs)

    # Add richer metadata per chunk
    for i, d in enumerate(split_docs):
        d.metadata["doc_id"] = d.metadata.get("doc_id") or str(uuid4())
        d.metadata["chunk_index"] = i
        d.metadata["collection"] = COLLECTION_NAME
        # Lightweight section guess to help filtering/reranking
        sec = guess_section_title(d.page_content)
        if sec:
            d.metadata["section_guess"] = sec

    return split_docs


# -------------------------------
# 5) Build Milvus vector store
# -------------------------------
def build_milvus_store(docs: List, embeddings) -> Milvus:
    """
    Creates/uses a Milvus collection with a reasonable index.
    For all-MiniLM-L6-v2 (384-dim), normalized embeddings + IP metric is common.
    """
    connection_args = {"host": HOST, "port": PORT}
    # You can override default index params if desired:
    #   metric_type: "IP" or "L2" (use IP with normalized vectors)
    #   index_type: IVF_FLAT / IVF_SQ8 / HNSW etc. (HNSW/IVF good starting points)
    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    search_params = {"metric_type": "IP", "params": {"ef": 64}}

    vs = Milvus.from_documents(
        docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_args=connection_args,
        # vector_field="embedding",  # uncomment if you’ve customized your schema
        index_params=index_params,
        search_params=search_params,
        # auto_id=True,              # let Milvus assign primary keys
        # drop_old=False             # keep existing data if the collection exists
    )
    return vs


# -------------------------------
# 6) Optional: Hybrid & Reranking
# -------------------------------
def build_hybrid_retriever(vectorstore: Milvus, documents_for_bm25: List):
    """
    Returns an EnsembleRetriever combining dense vector search with BM25.
    Only if BM25 and Ensemble retrievers are available.
    """
    if not (HAS_BM25 and HAS_ENSEMBLE):
        return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    bm25 = BM25Retriever.from_documents(documents_for_bm25)
    bm25.k = TOP_K

    dense = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    hybrid = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.35, 0.65],  # tune weights to your domain
    )
    return hybrid

def cross_encoder_rerank(query: str, docs: List):
    """
    Re-rank retrieved docs using a cross-encoder for higher precision.
    Requires sentence-transformers CrossEncoder installed.
    """
    if not HAS_CROSS or not docs:
        return docs
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    # Attach and sort
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked]


# -------------------------------
# 7) Demo query & sanity checks
# -------------------------------
def demo_query(retriever, query: str, rerank: bool = False):
    # retrieve candidates
    docs = retriever.get_relevant_documents(query)

    if rerank:
        docs = cross_encoder_rerank(query, docs)

    print(f"\n=== Query: {query} ===")
    for i, d in enumerate(docs[:TOP_K], 1):
        meta = d.metadata or {}
        page = meta.get("page", "N/A")
        sec = meta.get("section_guess", "")
        print(f"\n[{i}] Page {page}  {('— ' + sec) if sec else ''}")
        print(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
        print(f"Meta: {{source: {meta.get('source')}, chunk_index: {meta.get('chunk_index')}}}")


# -------------------------------
# 8) Main
# -------------------------------
def main():
    print("Loading and preparing documents...")
    docs = load_and_prepare_documents(PDF_PATH)

    print("Building embeddings...")
    embeddings = build_embeddings()

    print("Creating Milvus vector store (this may take a bit on first run)...")
    vstore = build_milvus_store(docs, embeddings)

    # Choose your retriever:
    retriever = build_hybrid_retriever(vstore, docs)  # falls back to dense-only if BM25 not installed

    # Demo: run a few queries that matter for your spec
    sample_queries = [
        "asphalt paving compaction requirements",
        "concrete curing period and temperature",
        "traffic control devices specifications",
        "aggregate gradation for base course",
    ]
    for q in sample_queries:
        demo_query(retriever, q, rerank=True)  # set rerank=False to skip cross-encoder

    print("\n✅ PDF chunks embedded and stored in Milvus; retrieval sanity-checks printed.")

if __name__ == "__main__":
    main()
