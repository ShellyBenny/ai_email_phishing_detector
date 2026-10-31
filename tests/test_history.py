"""
Tests for SQLite History Logging and Feedback Module (src/history.py).
"""

from pathlib import Path
import pytest
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.history import (
    clear_history,
    get_connection,
    get_history,
    get_history_stats,
    log_scan,
    record_feedback,
)


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary database path for isolated testing."""
    return tmp_path / "test_audit.db"


def test_log_scan_and_get_history(temp_db):
    scan_id = log_scan(
        email_snippet="Urgent: your account credentials must be verified",
        prediction="Phishing",
        confidence=98.7,
        risk_score=88.5,
        risk_level="High",
        model_used="Linear SVM",
        db_path=temp_db,
    )
    assert scan_id > 0

    df = get_history(db_path=temp_db)
    assert len(df) == 1
    assert df.iloc[0]["prediction"] == "Phishing"
    assert df.iloc[0]["risk_level"] == "High"
    assert df.iloc[0]["model_used"] == "Linear SVM"


def test_history_filtering_by_label(temp_db):
    log_scan("Legit email 1", "Legitimate", 95.0, 10.0, "Low", db_path=temp_db)
    log_scan("Spam email 1", "Spam", 88.0, 45.0, "Medium", db_path=temp_db)
    log_scan("Phishing email 1", "Phishing", 99.0, 92.0, "High", db_path=temp_db)

    df_all = get_history(label_filter="All", db_path=temp_db)
    assert len(df_all) == 3

    df_phish = get_history(label_filter="Phishing", db_path=temp_db)
    assert len(df_phish) == 1
    assert df_phish.iloc[0]["prediction"] == "Phishing"

    df_legit = get_history(label_filter="Legitimate", db_path=temp_db)
    assert len(df_legit) == 1
    assert df_legit.iloc[0]["prediction"] == "Legitimate"


def test_record_feedback_and_stats(temp_db):
    scan_id = log_scan("Test email for feedback", "Spam", 91.0, 50.0, "Medium", db_path=temp_db)
    success = record_feedback(scan_id, is_correct=True, notes="Confirmed spam", db_path=temp_db)
    assert success is True

    df = get_history(db_path=temp_db)
    assert df.iloc[0]["feedback"] == "Correct"

    stats = get_history_stats(db_path=temp_db)
    assert stats["total_scans"] == 1
    assert stats["counts_by_label"]["Spam"] == 1
    assert stats["feedback_counts"]["Correct"] == 1


def test_clear_history(temp_db):
    log_scan("Sample email", "Legitimate", 90.0, 15.0, "Low", db_path=temp_db)
    assert len(get_history(db_path=temp_db)) == 1

    cleared = clear_history(db_path=temp_db)
    assert cleared is True
    assert len(get_history(db_path=temp_db)) == 0

