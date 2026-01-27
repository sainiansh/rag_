from sentence_transformers import SentenceTransformer
import PyPDF2
from database import add_document, add_chunk
import pinecone

model = SentenceTransformer("all-MiniLM-L6-v2")

pinecone.init(api_key="YOUR_KEY", environment="YOUR_ENV")
index = pinecone.Index("rag-index")

def read_file(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join([p.extract_text() for p in reader.pages])
    return file.read().decode()

def chunk_text(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

def ingest_files(files):
    for file in files:
        text = read_file(file)
        doc_id = add_document(file.name)
        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()
        vectors = []

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = add_chunk(doc_id, chunk)
            vectors.append((str(chunk_id), emb, {"doc_id": doc_id}))

        index.upsert(vectors)
