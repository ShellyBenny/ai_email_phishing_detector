"""
SQLite-based Scan History and User Feedback Logger
for AI-Based Email Phishing Detection.

Stores email scan audit trails and user feedback verdicts:
- id: primary key
- timestamp: ISO 8601 string
- email_snippet: First 150 characters of email
- prediction: Legitimate, Spam, or Phishing
- confidence: Model confidence percentage (0-100)
- risk_score: Overall risk score (0-100)
- risk_level: Low, Medium, or High
- model_used: Name of the model
- feedback: 'Correct', 'Incorrect', or NULL
- feedback_notes: Optional user comments
"""

from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "history.db"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create SQLite connection and ensure tables exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection):
    """Initialize database schema with proper indices."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            email_snippet TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            model_used TEXT NOT NULL,
            feedback TEXT DEFAULT NULL,
            feedback_notes TEXT DEFAULT NULL
        );
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp
        ON scan_history (timestamp DESC);
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scan_history_prediction
        ON scan_history (prediction);
        """
    )
    conn.commit()


def log_scan(
    email_snippet: str,
    prediction: str,
    confidence: float,
    risk_score: float,
    risk_level: str,
    model_used: str = "Best Model",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """
    Log an email scan result into SQLite.

    Returns:
        int: The inserted record ID.
    """
    clean_snippet = " ".join(str(email_snippet).split())[:180]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scan_history (
                timestamp, email_snippet, prediction, confidence,
                risk_score, risk_level, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                clean_snippet,
                str(prediction),
                round(float(confidence), 2),
                round(float(risk_score), 1),
                str(risk_level),
                str(model_used),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def record_feedback(
    scan_id: int,
    is_correct: bool,
    notes: str = "",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Record user feedback ('Correct' or 'Incorrect') for a given scan."""
    feedback_str = "Correct" if is_correct else "Incorrect"
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE scan_history
            SET feedback = ?, feedback_notes = ?
            WHERE id = ?
            """,
            (feedback_str, str(notes), int(scan_id)),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_history(
    limit: int = 100,
    label_filter: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Retrieve scan history as a pandas DataFrame.

    Args:
        limit (int): Maximum records to fetch.
        label_filter (str): Optional filter ('Legitimate', 'Spam', 'Phishing').
    """
    conn = get_connection(db_path)
    try:
        if label_filter and label_filter != "All":
            query = """
                SELECT id, timestamp, email_snippet, prediction, confidence,
                       risk_score, risk_level, model_used, feedback
                FROM scan_history
                WHERE prediction = ?
                ORDER BY id DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(label_filter, limit))
        else:
            query = """
                SELECT id, timestamp, email_snippet, prediction, confidence,
                       risk_score, risk_level, model_used, feedback
                FROM scan_history
                ORDER BY id DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    finally:
        conn.close()


def clear_history(db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    """Delete all records from scan_history."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scan_history")
        conn.commit()
        return True
    finally:
        conn.close()


def get_history_stats(db_path: Path | str = DEFAULT_DB_PATH) -> dict:
    """Get aggregate statistics for the history dashboard."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scan_history")
        total_scans = cursor.fetchone()[0]

        cursor.execute("SELECT prediction, COUNT(*) FROM scan_history GROUP BY prediction")
        counts_by_label = dict(cursor.fetchall())

        cursor.execute("SELECT feedback, COUNT(*) FROM scan_history WHERE feedback IS NOT NULL GROUP BY feedback")
        feedback_counts = dict(cursor.fetchall())

        return {
            "total_scans": total_scans,
            "counts_by_label": counts_by_label,
            "feedback_counts": feedback_counts,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    test_db = BASE_DIR / "data" / "test_history.db"
    scan_id = log_scan(
        email_snippet="Urgent invoice attached for review",
        prediction="Phishing",
        confidence=98.5,
        risk_score=85.0,
        risk_level="High",
        model_used="Best Model",
        db_path=test_db,
    )
    print(f"Logged scan id: {scan_id}")
    record_feedback(scan_id, is_correct=True, db_path=test_db)
    df = get_history(db_path=test_db)
    print("History record:")
    print(df)
    clear_history(db_path=test_db)
    if test_db.exists():
        test_db.unlink()
    print("Test passed and test DB cleaned up.")

