import os
import streamlit as st
from pinecone import Pinecone
import PyPDF2
from sentence_transformers import SentenceTransformer
from database import add_document, add_chunk

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

@st.cache_resource
def load_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index("rag-index")

index = load_index()

def read_file(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    return file.read().decode("utf-8")

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def ingest_files(files):
    for file in files:
        text = read_file(file)
        doc_id = add_document(file.name, file.name.split(".")[-1])
        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()
        vectors = []

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = add_chunk(doc_id, chunk)
            vectors.append(
                (str(chunk_id), emb, {"doc_id": doc_id})
            )

        index.upsert(vectors=vectors)
