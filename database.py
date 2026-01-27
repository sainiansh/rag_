import sqlite3
from datetime import datetime

conn = sqlite3.connect("data/rag.db", check_same_thread=False)
cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        uploaded_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER,
        chunk_text TEXT,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS retrieval_logs (
        query TEXT,
        chunk_ids TEXT,
        score REAL,
        timestamp TEXT
    )
    """)

    conn.commit()

def add_document(file_name):
    cursor.execute(
        "INSERT INTO documents VALUES (NULL, ?, ?)",
        (file_name, datetime.now())
    )
    conn.commit()
    return cursor.lastrowid

def add_chunk(doc_id, text):
    cursor.execute(
        "INSERT INTO chunks VALUES (NULL, ?, ?)",
        (doc_id, text)
    )
    conn.commit()
    return cursor.lastrowid

def delete_document(doc_id):
    cursor.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    conn.commit()

def get_documents():
    cursor.execute("SELECT * FROM documents")
    return cursor.fetchall()
