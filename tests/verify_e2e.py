"""
End-to-End System Integration Test.
Verifies all functional flows of the application:
1. Three sample email predictions (Legitimate, Spam, Phishing)
2. .eml file upload and header parsing
3. Batch CSV scanning and metrics computation
4. PDF report generation and binary integrity
5. SQLite scan history and feedback recording
"""

import io
from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.history import clear_history, get_history, get_history_stats, log_scan, record_feedback
from src.predict import PhishingPredictor
from src.report import generate_pdf_report


def run_e2e_tests():
    print("=" * 70)
    print("           END-TO-END VERIFICATION OF PHASES 1 THROUGH 7")
    print("=" * 70)

    predictor = PhishingPredictor(models_dir=BASE_DIR / "models")
    print("[1/5] Testing 3 Sample Email Classifications:")

    samples = {
        "Legitimate": """From: aris.thorne@university-campus.edu
Subject: Biology 301: Assignment 2 Feedback & Office Hours
Dear Students, I have posted grades and comments for Assignment 2 on the course portal. Office hours are Tuesday 2 PM.""",
        "Spam": """From: deals@newsletter-shopping.com
Subject: Exclusive Online Clearance: Save 70% Today Only!
Big savings on vitamins, wellness supplements, and everyday health essentials! Visit our catalog today to claim your discount code.""",
        "Phishing": """From: "PayPal Account Security" <alert@paypa1-security-verification.xyz>
Subject: URGENT: Your PayPal Account Has Been Suspended!
We detected multiple unauthorized logins. Verify your identity immediately or account will be deleted: http://192.168.1.100/login""",
    }

    results = {}
    for name, text in samples.items():
        res = predictor.predict(text, model_name="Best Model")
        results[name] = res
        print(f"  - {name:<10} -> Predicted: {res['prediction']:<10} (Conf: {res['confidence_score']:5.1f}%, Risk: {res['risk_score']:4.1f}/100 [{res['risk_level']}])")
        assert res["prediction"] in {"Legitimate", "Spam", "Phishing"}
        assert 0.0 <= res["confidence_score"] <= 100.0
        assert 0.0 <= res["risk_score"] <= 100.0

    print("\n[2/5] Testing .EML File Processing with Full RFC 5322 Headers:")
    eml_content = """From: "Bank Security Alert" <support@fake-bank-desk.xyz>
Reply-To: phisher-inbox@mail-relay.ru
Return-Path: bounce@marketing-blast.xyz
Subject: Urgent Security Warning: Card Compromised
Date: Fri, 30 Oct 2026 21:00:00 +0000
Authentication-Results: mx.google.com; spf=fail; dkim=fail; dmarc=fail

Dear customer, unauthorized transaction detected. Click http://192.168.1.50 to restore card.
"""
    eml_res = predictor.predict(eml_content, model_name="Best Model")
    print(f"  - .EML Prediction: {eml_res['prediction']} (Risk Score: {eml_res['risk_score']}/100 [{eml_res['risk_level']}])")
    assert eml_res["header_analysis"]["headers_present"] is True
    assert len(eml_res["header_analysis"]["red_flags"]) >= 3
    print(f"  - Header Red Flags Detected: {len(eml_res['header_analysis']['red_flags'])}")

    print("\n[3/5] Testing Batch CSV Classification Flow:")
    batch_csv_data = """id,email_text
1,"Hi team, please find the meeting notes for sprint planning attached. Thanks, Alex"
2,"Lowest price online prescription drugs! 80% discount clearance. Order today!"
3,"URGENT: Your bank account will be suspended within 24 hours. Verify now: http://192.168.1.1/login"
"""
    batch_df = pd.read_csv(io.StringIO(batch_csv_data))
    batch_results = []
    for _, row in batch_df.iterrows():
        b_res = predictor.predict(row["email_text"], model_name="Best Model")
        batch_results.append({
            "id": row["id"],
            "pred": b_res["prediction"],
            "risk": b_res["risk_score"],
        })
    b_out_df = pd.DataFrame(batch_results)
    print(f"  - Batch classified {len(b_out_df)} records successfully:")
    for _, r in b_out_df.iterrows():
        print(f"      Row {r['id']}: {r['pred']} (Risk: {r['risk']})")
    assert len(b_out_df) == 3

    print("\n[4/5] Testing PDF Report Export:")
    pdf_bytes = generate_pdf_report(results["Phishing"], samples["Phishing"])
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    print(f"  - PDF Report successfully generated ({len(pdf_bytes):,} bytes, valid %PDF- header).")

    print("\n[5/5] Testing SQLite Scan History Audit Trail & Feedback:")
    test_db = BASE_DIR / "data" / "e2e_history_test.db"
    if test_db.exists():
        test_db.unlink()

    rec_id = log_scan(
        email_snippet="E2E test phishing scan",
        prediction="Phishing",
        confidence=99.1,
        risk_score=85.0,
        risk_level="High",
        model_used="Best Model",
        db_path=test_db,
    )
    assert rec_id == 1
    feedback_ok = record_feedback(rec_id, is_correct=True, notes="Verified in E2E", db_path=test_db)
    assert feedback_ok is True
    hist_df = get_history(db_path=test_db)
    assert len(hist_df) == 1
    assert hist_df.iloc[0]["feedback"] == "Correct"
    clear_ok = clear_history(db_path=test_db)
    assert clear_ok is True
    assert len(get_history(db_path=test_db)) == 0
    if test_db.exists():
        test_db.unlink()
    print("  - SQLite scan history, feedback logging, and clear history verified.")

    print("\n" + "=" * 70)
    print("           ALL END-TO-END FLOWS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_tests()
