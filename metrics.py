from datetime import datetime
from database import get_connection

def log_retrieval(query, chunk_ids, avg_score, latency_ms):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
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
    conn.close()
