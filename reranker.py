import streamlit as st
from sentence_transformers import CrossEncoder

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

def rerank(query, texts):
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:5]]
