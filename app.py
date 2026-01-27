import streamlit as st
from ingest import ingest_files
from retrieval import semantic_search
from reranker import rerank
from prompt import grounded_prompt
from database import init_db, get_documents, delete_document
from metrics import log_retrieval
from groq import Groq

client = Groq(api_key="YOUR_GROQ_KEY")

st.set_page_config(page_title="RAG System", layout="wide")
init_db()

st.title("📚 Production RAG System")

# Upload
files = st.file_uploader("Upload documents", accept_multiple_files=True)
if st.button("Ingest"):
    ingest_files(files)
    st.success("Documents ingested")

# Document management
st.subheader("📂 Documents")
docs = get_documents()
for d in docs:
    if st.button(f"Delete {d[1]}"):
        delete_document(d[0])
        st.experimental_rerun()

# Query
query = st.text_input("Ask a question")
if st.button("Search"):
    results = semantic_search(query)
    chunks = [r["metadata"]["doc_id"] for r in results]

    reranked = rerank(query, [r["id"] for r in results])
    context = "\n".join(reranked)

    prompt = grounded_prompt(context, query)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write(response.choices[0].message.content)
    log_retrieval(query, reranked, results[0]["score"])
