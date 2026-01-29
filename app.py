import time
import streamlit as st
import os
from groq import Groq

from database import init_db, get_documents, delete_document, get_chunks_by_ids
from ingest import ingest_files
from retrieval import hybrid_search
from reranker import rerank
from prompt import grounded_prompt
from metrics import log_retrieval

init_db()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 Production-Ready RAG System")

# Upload
files = st.file_uploader(
    "Upload documents (PDF/TXT)",
    accept_multiple_files=True
)

if st.button("Ingest Documents"):
    if not files:
        st.warning("Please upload at least one file.")
    else:
        ingest_files(files)
        st.success("Documents ingested successfully.")

# Documents
st.subheader("📂 Documents")
docs = get_documents()
for d in docs:
    col1, col2 = st.columns([4, 1])
    col1.write(d[1])
    if col2.button("Delete", key=d[0]):
        delete_document(d[0])
        st.experimental_rerun()

# Query
st.subheader("🔍 Ask a Question")
query = st.text_input("Enter your question")

if st.button("Search"):
    start = time.time()

    results = hybrid_search(query)
    chunk_ids = [r[0] for r in results[:8]]
    scores = [r[1] for r in results[:8]]

    texts = get_chunks_by_ids(chunk_ids)
    reranked_texts = rerank(query, texts)

    context = "\n\n".join(reranked_texts)
    prompt = grounded_prompt(context, query)

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    latency = (time.time() - start) * 1000
    log_retrieval(query, chunk_ids, sum(scores)/len(scores), latency)

    st.markdown("### ✅ Answer")
    st.write(response.choices[0].message.content)
