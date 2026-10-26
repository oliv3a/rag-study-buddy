import sqlite3
import uuid
from datetime import datetime

DB_FILE = "progress.db"


def connect():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            topic TEXT,
            question TEXT,
            correct INTEGER,      -- 1 = right, 0 = wrong
            timestamp TEXT
        )
    """)
    return conn


def new_session_id():
    """A random unique ID to group one study session's attempts."""
    return str(uuid.uuid4())[:8]


def log_attempt(session_id, topic, question, correct):
    with connect() as conn:
        conn.execute(
            "INSERT INTO attempts (session_id, topic, question, correct, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, topic, question, int(correct), datetime.now().isoformat(timespec="seconds")),
        )


def topic_stats():
    """Return [(topic, attempts, correct), ...], worst accuracy first."""
    with connect() as conn:
        return conn.execute("""
            SELECT topic, COUNT(*), SUM(correct)
            FROM attempts
            GROUP BY topic
            ORDER BY 1.0 * SUM(correct) / COUNT(*) ASC
        """).fetchall()


if __name__ == "__main__":
    # Quick self-test with fake data
    sid = new_session_id()
    log_attempt(sid, "dijkstra", "test q1", False)
    log_attempt(sid, "dijkstra", "test q2", False)
    log_attempt(sid, "sorting-quicksort", "test q3", True)
    for topic, total, right in topic_stats():
        print(f"{topic}: {right}/{total} correct ({100 * right // total}%)")