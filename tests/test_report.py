"""
Tests for PDF Threat Report Generation (src/report.py).
"""

from pathlib import Path
import pytest
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.report import generate_pdf_report, sanitize_text


def test_sanitize_text():
    raw = "Alert: ‘Account Suspended’ — verify password… 🚨"
    sanitized = sanitize_text(raw)
    assert "[ALERT]" in sanitized
    assert "Account Suspended" in sanitized
    assert isinstance(sanitized, str)


def test_generate_pdf_report_phishing():
    mock_scan_res = {
        "prediction": "Phishing",
        "confidence_score": 99.4,
        "probabilities": {"Legitimate": 0.001, "Spam": 0.005, "Phishing": 0.994},
        "model_used": "Linear SVM",
        "risk_score": 88.0,
        "risk_level": "High",
        "risk_reasons": [
            "AI Model detected high probability of phishing intent (99.4%).",
            "URL Red Flag: Registered under high-risk TLD (.xyz)",
            "Psychological Bait (Urgency): detected 'urgent', 'immediately'",
        ],
        "top_words": [
            {"word": "account", "weight": 0.38, "tfidf": 0.25},
            {"word": "verify", "weight": 0.22, "tfidf": 0.18},
        ],
        "header_analysis": {
            "headers_present": True,
            "sender_info": {"from_raw": "Security <sec@fake.xyz>", "reply_to_email": "drop@hack.ru"},
            "auth_results": {"spf": "fail", "dkim": "fail", "dmarc": "fail"},
            "red_flags": ["From vs Reply-To mismatch"],
        },
        "safety_tips": {
            "title": "Action Required: High Risk Phishing Email",
            "steps": ["Do not click any links.", "Report email to IT."],
        },
    }
    raw_email = "From: sec@fake.xyz\nSubject: Account Alert\n\nVerify immediately: http://fake.xyz"

    pdf_bytes = generate_pdf_report(mock_scan_res, raw_email)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    # PDF magic bytes
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_report_without_headers():
    mock_res = {
        "prediction": "Legitimate",
        "confidence_score": 97.2,
        "probabilities": {"Legitimate": 0.972, "Spam": 0.015, "Phishing": 0.013},
        "model_used": "Logistic Regression",
        "risk_score": 12.0,
        "risk_level": "Low",
        "risk_reasons": ["No active red flags detected."],
        "top_words": [{"word": "meeting", "weight": 0.25, "tfidf": 0.30}],
        "header_analysis": {"headers_present": False},
        "safety_tips": {
            "title": "Legitimate Communication",
            "steps": ["Email matches benign patterns."],
        },
    }
    pdf_bytes = generate_pdf_report(mock_res, "Hi team, see you at the sync meeting.")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")

