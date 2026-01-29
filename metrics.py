from database import conn
from datetime import datetime

def log_retrieval(query, chunk_ids, avg_score, latency_ms):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO retrieval_logs VALUES (NULL, ?, ?, ?, ?, ?)",
        (
            query,
            ",".join(map(str, chunk_ids)),
            avg_score,
            latency_ms,
            datetime.now()
        )
    )
    conn.commit()
