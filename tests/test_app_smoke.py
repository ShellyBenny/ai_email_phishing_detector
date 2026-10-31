"""
Smoke Tests for Streamlit App (app.py).
Verifies that app.py imports, defines required functions, and initializes cleanly.
"""

import ast
from pathlib import Path
import pytest
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
APP_PATH = BASE_DIR / "app.py"


def test_app_syntax_and_ast_compilation():
    """Verify app.py compiles cleanly without Python syntax errors."""
    assert APP_PATH.exists(), f"app.py not found at {APP_PATH}"
    source = APP_PATH.read_text(encoding="utf-8")
    parsed_ast = ast.parse(source)
    assert parsed_ast is not None


def test_app_critical_dependencies_available():
    """Ensure all required libraries used by app.py are present in the environment."""
    import streamlit
    import plotly
    import pandas
    import numpy
    import fpdf
    from src.predict import PhishingPredictor
    from src.history import log_scan
    from src.report import generate_pdf_report

    assert streamlit.__version__ is not None
    assert plotly.__version__ is not None
    assert fpdf.__version__ is not None


def test_predictor_loads_in_app_context():
    """Ensure the predictor loads the models successfully as expected by app.py."""
    from src.predict import PhishingPredictor
    p = PhishingPredictor(models_dir=BASE_DIR / "models")
    assert len(p.get_available_models()) >= 4
    assert p.vectorizer is not None


def test_parse_eml_payload_40kb_sample():
    """Ensure .eml parser correctly parses multipart HTML emails and extracts text."""
    from app import parse_eml_payload
    eml_path = BASE_DIR / "tests" / "sample_40kb_phishing.eml"
    assert eml_path.exists(), f"Sample .eml not found at {eml_path}"
    file_bytes = eml_path.read_bytes()
    assert len(file_bytes) > 25000  # Verify it is ~40KB

    parsed = parse_eml_payload(file_bytes)
    assert "Chase" in parsed["subject"] or "Alert" in parsed["subject"]
    assert "chase" in parsed["from"].lower()
    assert len(parsed["body_text"]) > 100
    assert "<html>" not in parsed["body_text"]  # HTML cleaned to readable text


