import sqlite3
from datetime import datetime
import os

DB_PATH = "data/rag.db"
os.makedirs("data", exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        file_type TEXT,
        uploaded_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER,
        chunk_text TEXT NOT NULL,
        page_number INTEGER,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS retrieval_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        retrieved_chunk_ids TEXT,
        avg_score REAL,
        latency_ms REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

def add_document(file_name, file_type):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO documents VALUES (NULL, ?, ?, ?)",
        (file_name, file_type, datetime.now())
    )

    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id

def add_chunk(doc_id, text, page_number=None):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chunks VALUES (NULL, ?, ?, ?)",
        (doc_id, text, page_number)
    )

    conn.commit()
    chunk_id = cur.lastrowid
    conn.close()
    return chunk_id

def get_documents():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM documents")
    rows = cur.fetchall()

    conn.close()
    return rows

def get_chunks_by_ids(chunk_ids):
    if not chunk_ids:
        return []

    conn = get_connection()
    cur = conn.cursor()

    placeholders = ",".join("?" * len(chunk_ids))
    cur.execute(
        f"SELECT chunk_text FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids
    )

    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

def delete_document(doc_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    cur.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    conn.commit()
    conn.close()
