import streamlit as st
import os
import io
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.milvus import Milvus
from PIL import Image
import PyPDF2
from docx import Document
import speech_recognition as sr
import google.generativeai as genai

load_dotenv()

# Milvus config
host = os.getenv('HOST', 'localhost')
port = os.getenv('PORT', '19530')
collection_name = os.getenv('MILVUS_COLLECTION', 'wyodotspecs')
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})

# Initialize Milvus vector store
vectorstore = Milvus(
    embeddings,
    connection_args={'host': host, 'port': port},
    collection_name=collection_name,
    vector_field="embedding"
)

st.title("WYDOT Multimodal RAG Chatbot (Milvus + Gemini)")

query = st.text_area("Enter your question or text:")

uploaded_files = st.file_uploader(
    "Upload files (image, audio, video, PDF, DOCX)", accept_multiple_files=True
)

extracted_texts = []
uploads = []
if uploaded_files:
    recognizer = sr.Recognizer()
    for f in uploaded_files:
        filename = (f.name or "").lower()
        raw = f.read()
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
            uploads.append({"bytes": raw, "mime": "image/*"})
        elif filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            uploads.append({"bytes": raw, "mime": "audio/*"})
            try:
                with sr.AudioFile(io.BytesIO(raw)) as source:
                    audio_data = recognizer.record(source)
                stt_text = recognizer.recognize_google(audio_data)
                if stt_text:
                    extracted_texts.append(stt_text)
            except Exception:
                pass
        elif filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            uploads.append({"bytes": raw, "mime": "video/*"})
        elif filename.endswith(".pdf"):
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() or ""
                if pdf_text.strip():
                    extracted_texts.append(pdf_text)
            except Exception:
                pass
        elif filename.endswith(".docx"):
            try:
                doc = Document(io.BytesIO(raw))
                doc_text = "\n".join(p.text for p in doc.paragraphs)
                if doc_text.strip():
                    extracted_texts.append(doc_text)
            except Exception:
                pass
        else:
            uploads.append({"bytes": raw, "mime": "application/octet-stream"})

full_text = query + "\n\n" + "\n\n".join(extracted_texts)

# RAG retrieval from Milvus
if st.button("Ask WYDOT Chatbot"):
    if full_text.strip():
        docs = vectorstore.similarity_search(full_text, k=5)
        context_text = "\n\n".join(d.page_content for d in docs if d.page_content)
        sources = [
            {
                "page": d.metadata.get("page"),
                "source": d.metadata.get("source") or d.metadata.get("file"),
                "preview": (d.page_content or "")[:300],
            }
            for d in docs
        ]
        # Build Gemini parts
        parts = []
        parts.append({"text": "You are WYDOT’s helpful assistant. Answer ONLY using the provided context (vectorstore + uploaded materials). Ensure clarity, brevity, and human-like responses. If the answer is not contained, answer it based on your role. Be concise and clear."})
        if context_text:
            parts.append({"text": f"CONTEXT (from vectorstore):\n{context_text}"})
        if full_text.strip():
            parts.append({"text": f"UPLOADED DOC/TEXT CONTEXT:\n{full_text}"})
        if uploads:
            for item in uploads:
                b = item.get("bytes")
                mime = item.get("mime") or "application/octet-stream"
                if not b:
                    continue
                parts.append({"inline_data": {"mime_type": mime, "data": b}})
        if query:
            parts.append({"text": f"QUESTION:\n{query}"})
        # Gemini call
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(parts)
        st.markdown("### WYDOT Chatbot Response:")
        st.write(resp.text)
        st.markdown("---")
        st.markdown("### Top RAG Sources:")
        for i, src in enumerate(sources):
            st.markdown(f"**Source {i+1}:** Page {src['page']} | {src['source']}\nPreview: {src['preview']}")
    else:
        st.warning("Please enter a question or upload files with text.")
