import os
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from database import get_chunks_by_ids

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

@st.cache_resource
def load_index():
    return pinecone.Index("rag-index")

index = load_index()

def semantic_search(query, top_k=10):
    q_emb = model.encode(query).tolist()
    results = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]

def hybrid_search(query, top_k=10):
    semantic_results = semantic_search(query, top_k)
    chunk_ids = [int(r["id"]) for r in semantic_results]
    texts = get_chunks_by_ids(chunk_ids)

    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())

    combined = []
    for i, r in enumerate(semantic_results):
        score = 0.7 * r["score"] + 0.3 * bm25_scores[i]
        combined.append((int(r["id"]), score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined
