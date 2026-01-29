import sqlite3
from datetime import datetime
import os

DB_PATH = "data/rag.db"
os.makedirs("data", exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        file_type TEXT,
        uploaded_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER,
        chunk_text TEXT NOT NULL,
        page_number INTEGER,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
    """)

    cursor.execute("""
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

def add_document(file_name, file_type):
    cursor.execute(
        "INSERT INTO documents VALUES (NULL, ?, ?, ?)",
        (file_name, file_type, datetime.now())
    )
    conn.commit()
    return cursor.lastrowid

def add_chunk(doc_id, text, page_number=None):
    cursor.execute(
        "INSERT INTO chunks VALUES (NULL, ?, ?, ?)",
        (doc_id, text, page_number)
    )
    conn.commit()
    return cursor.lastrowid

def get_documents():
    cursor.execute("SELECT * FROM documents")
    return cursor.fetchall()

def get_chunks_by_ids(chunk_ids):
    placeholders = ",".join("?" * len(chunk_ids))
    cursor.execute(
        f"SELECT chunk_text FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids
    )
    return [row[0] for row in cursor.fetchall()]

def delete_document(doc_id):
    cursor.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    conn.commit()
