"""
Honest Evaluation Script for 15 Fresh, Unseen, Realistic Test Emails (Phase 7).
Runs the trained detection pipeline on:
- 5 Legitimate emails
- 5 Spam emails
- 5 Modern-style Phishing emails (package delivery, bank alert, fake job, M365 expiry, IRS tax refund)

Reports all predictions, probabilities, risk scores, and failure modes honestly.
"""

from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import PhishingPredictor

SAMPLES_CSV = BASE_DIR / "tests" / "fresh_samples.csv"


def evaluate_fresh_samples():
    print("=" * 80)
    print("       HONEST EVALUATION ON 15 UNSEEN REALISTIC EMAILS (PHASE 7)")
    print("=" * 80)

    if not SAMPLES_CSV.exists():
        raise FileNotFoundError(f"Missing fresh samples file at {SAMPLES_CSV}")

    df = pd.read_csv(SAMPLES_CSV)
    predictor = PhishingPredictor(models_dir=BASE_DIR / "models")

    results = []
    correct_count = 0
    total = len(df)

    print(f"\nEvaluating {total} unseen samples across Legitimate, Spam, and Phishing...\n")

    for _, row in df.iterrows():
        sample_id = row["id"]
        category = row["category"]
        ground_truth = row["ground_truth"]
        subject = row["subject"]
        body = row["body"]

        full_email = f"Subject: {subject}\n\n{body}"
        res = predictor.predict(full_email, model_name="Best Model")

        pred = res["prediction"]
        conf = res["confidence_score"]
        risk = res["risk_score"]
        risk_lvl = res["risk_level"]
        is_match = (pred == ground_truth)

        if is_match:
            correct_count += 1
            verdict_symbol = "✓ PASS"
        else:
            verdict_symbol = "✗ FAIL"

        results.append({
            "id": sample_id,
            "category": category,
            "ground_truth": ground_truth,
            "predicted": pred,
            "match": is_match,
            "confidence": conf,
            "risk_score": risk,
            "risk_level": risk_lvl,
            "subject": subject,
            "top_words": [w["word"] for w in res.get("top_words", [])[:3]],
            "red_flags_count": len(res.get("risk_reasons", [])),
        })

        print(f"[{verdict_symbol}] Sample #{sample_id:<2} ({category})")
        print(f"     Subject   : {subject}")
        print(f"     Truth     : {ground_truth:<10} | Predicted: {pred:<10} (Conf: {conf:5.1f}%, Risk: {risk:4.1f}/100 [{risk_lvl}])")
        if not is_match:
            print(f"     >>> MISCLASSIFICATION ANALYSIS: Model predicted '{pred}' instead of '{ground_truth}'.")
            print(f"         Probabilities: {res['probabilities']}")
            print(f"         Top Influential Words: {res.get('top_words', [])[:4]}")
            print(f"         Why this score: {res.get('risk_reasons', [])[:2]}")
        print("-" * 80)

    acc = (correct_count / total) * 100
    print("\n" + "=" * 80)
    print(f"  FRESH SAMPLES EVALUATION SUMMARY: {correct_count}/{total} CORRECT ({acc:.1f}% ACCURACY)")
    print("=" * 80)

    # Per-class accuracy
    res_df = pd.DataFrame(results)
    for cat in ["Legitimate", "Spam", "Phishing"]:
        cat_df = res_df[res_df["ground_truth"] == cat]
        cat_acc = (cat_df["match"].sum() / len(cat_df)) * 100
        print(f"  - {cat:<11} Accuracy: {cat_df['match'].sum()}/{len(cat_df)} ({cat_acc:5.1f}%)")

    print("\nDetailed breakdown saved for reporting.\n")
    return res_df


if __name__ == "__main__":
    evaluate_fresh_samples()

