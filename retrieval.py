from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pinecone

model = SentenceTransformer("all-MiniLM-L6-v2")

pinecone.init(api_key="YOUR_KEY", environment="YOUR_ENV")
index = pinecone.Index("rag-index")

def semantic_search(query, k=10):
    q_emb = model.encode(query).tolist()
    res = index.query(vector=q_emb, top_k=k, include_metadata=True)
    return res["matches"]

def hybrid_search(query, corpus_chunks, k=10):
    tokenized = [c.split() for c in corpus_chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())

    semantic = semantic_search(query, k)

    combined = []
    for i, s in enumerate(semantic):
        combined.append((s["id"], 0.7 * s["score"] + 0.3 * bm25_scores[i]))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]
