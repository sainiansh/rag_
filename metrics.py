from database import conn
from datetime import datetime

def log_retrieval(query, chunk_ids, score):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO retrieval_logs VALUES (?, ?, ?, ?)",
        (query, ",".join(chunk_ids), score, datetime.now())
    )
    conn.commit()
