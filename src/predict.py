"""
Prediction, Explainability, and Comprehensive Risk Scoring Engine
for AI-Based Email Phishing Detection.

Features:
1. Model inference with probability calibration across:
   - Best Model (Linear SVM calibrated)
   - Naive Bayes
   - Logistic Regression
   - Random Forest
2. Top 8 Influential Words / N-grams feature attribution with text highlighting
3. Rule-based URL, urgency, and attachment red flag extraction
4. Email header spoofing and SPF/DKIM verification
5. Transparent weighted Risk Score (0 - 100) with plain-English explanation list
6. Dynamic Safety Tips box tailored to predicted class
"""

import html
import json
from pathlib import Path
import re
import sys
import joblib
import numpy as np

# Set project paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.features import extract_all_features
from src.headers import analyze_headers
from src.preprocess import clean_text

MODELS_DIR = BASE_DIR / "models"
CLASS_NAMES = ["Legitimate", "Phishing", "Spam"]

# Weights for Risk Score Calculation (documented transparent formula)
WEIGHTS_WITH_HEADERS = {
    "model_phishing_prob": 0.40,
    "url_flags": 0.25,
    "urgency_flags": 0.15,
    "attachment_flags": 0.10,
    "header_flags": 0.10,
}

WEIGHTS_WITHOUT_HEADERS = {
    "model_phishing_prob": 0.45,
    "url_flags": 0.25,
    "urgency_flags": 0.20,
    "attachment_flags": 0.10,
    "header_flags": 0.00,
}


class PhishingPredictor:
    """Production predictor and explainer."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.vectorizer = None
        self.models = {}
        self.metrics = {}
        self.best_model_name = "Best Model"
        self._load_assets()

    def _load_assets(self):
        """Load TF-IDF vectorizer, models, and metrics."""
        vec_path = self.models_dir / "vectorizer.joblib"
        if not vec_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vec_path}. Run src/train.py first.")

        self.vectorizer = joblib.load(vec_path)

        model_files = {
            "Best Model": "best_model.joblib",
            "Linear SVM": "linear_svm.joblib",
            "Logistic Regression": "logistic_regression.joblib",
            "Naive Bayes": "naive_bayes.joblib",
            "Random Forest": "random_forest.joblib",
        }

        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                self.models[name] = joblib.load(path)

        metrics_path = self.models_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
                self.best_model_name = self.metrics.get("best_model_name", "Linear SVM")

    def get_available_models(self) -> list[str]:
        """Return list of loaded model names."""
        return list(self.models.keys())

    def _get_top_features_explanation(self, model, model_name: str, clean_email: str, predicted_class: str, top_k: int = 8) -> list[dict]:
        """
        Calculate the top influential words/phrases that pushed the prediction
        toward the predicted class using mathematical feature attribution.
        """
        if not clean_email or self.vectorizer is None:
            return []

        doc_vector = self.vectorizer.transform([clean_email])
        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = doc_vector.nonzero()[1]

        if len(nonzero_indices) == 0:
            return []

        # Find class index
        model_classes = list(getattr(model, "classes_", CLASS_NAMES))
        if predicted_class in model_classes:
            class_idx = model_classes.index(predicted_class)
        else:
            class_idx = 0

        contributions = []

        # 1. CalibratedClassifierCV (wrapping LinearSVC or other models)
        if hasattr(model, "calibrated_classifiers_"):
            base_clfs = model.calibrated_classifiers_
            weights = []
            for cc in base_clfs:
                estimator = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                if hasattr(estimator, "coef_"):
                    weights.append(estimator.coef_)
            if weights:
                avg_coef = np.mean(weights, axis=0)
                if len(avg_coef.shape) == 1:
                    w_class = avg_coef
                else:
                    w_class = avg_coef[class_idx]
                for idx in nonzero_indices:
                    tfidf_val = doc_vector[0, idx]
                    score = float(tfidf_val * w_class[idx])
                    contributions.append({
                        "word": feature_names[idx],
                        "weight": round(score, 4),
                        "tfidf": round(float(tfidf_val), 4),
                    })

        # 2. Logistic Regression or LinearSVC
        elif hasattr(model, "coef_"):
            coef = model.coef_
            w_class = coef[class_idx] if len(coef.shape) > 1 else coef
            for idx in nonzero_indices:
                tfidf_val = doc_vector[0, idx]
                score = float(tfidf_val * w_class[idx])
                contributions.append({
                    "word": feature_names[idx],
                    "weight": round(score, 4),
                    "tfidf": round(float(tfidf_val), 4),
                })

        # 3. Multinomial Naive Bayes
        elif hasattr(model, "feature_log_prob_"):
            log_probs = model.feature_log_prob_
            mean_log_prob = np.mean(log_probs, axis=0)
            class_log_prob = log_probs[class_idx]
            for idx in nonzero_indices:
                tfidf_val = doc_vector[0, idx]
                score = float(tfidf_val * (class_log_prob[idx] - mean_log_prob[idx]))
                contributions.append({
                    "word": feature_names[idx],
                    "weight": round(score, 4),
                    "tfidf": round(float(tfidf_val), 4),
                })

        # 4. Random Forest
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for idx in nonzero_indices:
                tfidf_val = doc_vector[0, idx]
                score = float(tfidf_val * importances[idx])
                contributions.append({
                    "word": feature_names[idx],
                    "weight": round(score, 4),
                    "tfidf": round(float(tfidf_val), 4),
                })

        # Fallback: TF-IDF values alone
        else:
            for idx in nonzero_indices:
                tfidf_val = doc_vector[0, idx]
                contributions.append({
                    "word": feature_names[idx],
                    "weight": round(float(tfidf_val), 4),
                    "tfidf": round(float(tfidf_val), 4),
                })

        # Sort descending by contribution weight
        contributions.sort(key=lambda x: x["weight"], reverse=True)
        return contributions[:top_k]

    def _highlight_text(self, raw_text: str, top_words: list[dict], predicted_class: str) -> str:
        """
        Generate HTML-highlighted snippet where the top influential words
        are wrapped inline in clean highlighted marks matching the light theme,
        rendered as normal readable paragraphs.
        """
        if not raw_text:
            return ""

        # For HTML-heavy emails, extract clean readable text preserving paragraphs
        if "<" in raw_text and ">" in raw_text:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw_text, "html.parser")
                for s in soup(["script", "style"]):
                    s.decompose()
                display_text = soup.get_text(separator="\n")
                display_text = re.sub(r"\n\s*\n+", "\n\n", display_text).strip()
            except Exception:
                display_text = raw_text
        else:
            display_text = raw_text

        # Cap at sensible limit for rendering performance if extremely huge
        if len(display_text) > 6000:
            display_text = display_text[:6000] + "\n\n... [Remaining text truncated for display] ..."

        color_map = {
            "Phishing": ("#FEE2E2", "#991B1B", "#FCA5A5"),  # light red bg, dark red text, border
            "Spam": ("#FEF3C7", "#92400E", "#FCD34D"),      # light amber bg, dark amber text, border
            "Legitimate": ("#DCFCE7", "#166534", "#86EFAC"),# light green bg, dark green text, border
        }
        bg, txt, border = color_map.get(predicted_class, ("#E0F2FE", "#0369A1", "#7DD3FC"))

        escaped = html.escape(display_text)
        words_to_highlight = [w["word"] for w in top_words if len(w["word"]) > 2]

        for w in words_to_highlight:
            pat = re.compile(rf"\b({re.escape(html.escape(w))})\b", re.IGNORECASE)
            replacement = (
                rf'<mark style="background-color: {bg}; color: {txt}; '
                rf'padding: 1px 5px; border-radius: 4px; font-weight: 600; '
                rf'border: 1px solid {border};">\1</mark>'
            )
            escaped = pat.sub(replacement, escaped)

        # Convert linebreaks to HTML paragraphs for natural reading
        paragraphs = escaped.split("\n\n")
        html_paragraphs = "".join([f"<p style='margin-bottom: 0.75rem;'>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip()])

        return html_paragraphs

    def calculate_risk_score(self, phishing_prob: float, features: dict, headers: dict) -> tuple[float, str, list[str]]:
        """
        Compute transparent weighted Risk Score (0 - 100) and explain reasons.
        """
        has_headers = headers.get("headers_present", False)
        weights = WEIGHTS_WITH_HEADERS if has_headers else WEIGHTS_WITHOUT_HEADERS

        reasons = []

        # 1. Model Phishing Component (0 to 100)
        model_score = phishing_prob * 100.0
        if model_score > 75:
            reasons.append(f"AI Model detected high probability of phishing intent ({model_score:.1f}%).")
        elif model_score > 40:
            reasons.append(f"AI Model detected moderate suspicion of malicious patterns ({model_score:.1f}%).")

        # 2. URL Component
        url_analysis = features.get("url_analysis", {})
        url_flags = url_analysis.get("red_flags", [])
        url_score = min(len(url_flags) * 30.0, 100.0)
        for flag in url_flags[:3]:
            reasons.append(f"URL Red Flag: {flag}")

        # 3. Urgency & Threat Component
        urgency = features.get("urgency_analysis", {})
        triggers = urgency.get("total_triggers", 0)
        cat_count = urgency.get("categories_triggered", 0)
        urgency_score = min(triggers * 20.0 + cat_count * 15.0, 100.0)
        for cat, phrases in urgency.get("breakdown", {}).items():
            sample_phrases = ", ".join([f"'{p}'" for p in phrases[:2]])
            reasons.append(f"Psychological Bait ({cat}): detected {sample_phrases}")

        # 4. Attachment Component
        att = features.get("attachment_analysis", {})
        att_flags = att.get("red_flags", [])
        att_score = min(len(att_flags) * 45.0, 100.0)
        for a_flag in att_flags:
            reasons.append(f"Attachment Warning: {a_flag}")

        # 5. Header Component
        if has_headers:
            header_penalty = float(headers.get("header_risk_penalty", 0.0))
            for h_flag in headers.get("red_flags", []):
                reasons.append(f"Email Header Red Flag: {h_flag}")
        else:
            header_penalty = 0.0

        # Weighted calculation
        total_risk = (
            weights["model_phishing_prob"] * model_score
            + weights["url_flags"] * url_score
            + weights["urgency_flags"] * urgency_score
            + weights["attachment_flags"] * att_score
            + weights["header_flags"] * header_penalty
        )

        total_risk = max(0.0, min(100.0, round(total_risk, 1)))

        # Determine Risk Level
        if total_risk >= 70.0:
            level = "High"
        elif total_risk >= 35.0:
            level = "Medium"
        else:
            level = "Low"

        if not reasons:
            reasons.append("No active red flags detected; email matches standard benign communications.")

        return total_risk, level, reasons

    def get_safety_tips(self, predicted_class: str, risk_level: str) -> dict:
        """Dynamic safety recommendations tailored to predictions."""
        if predicted_class == "Phishing" or risk_level == "High":
            return {
                "title": "🚨 Action Required: High Risk Phishing Email",
                "color": "danger",
                "steps": [
                    "DO NOT click any links, open attachments, or reply to this message.",
                    "If this email claims to be from a service you use (e.g., PayPal, Bank, IT Helpdesk), open a new browser tab and navigate directly to the official URL manually.",
                    "Never enter login credentials or multi-factor authentication (MFA) codes on pages reached through unsolicited emails.",
                    "Report this email to your organization's security operations team or abuse reporting inbox.",
                    "If you already clicked a link or entered credentials, immediately change your password from an uninfected device and notify IT.",
                ],
            }
        elif predicted_class == "Spam" or risk_level == "Medium":
            return {
                "title": "⚠️ Caution: Unsolicited Spam or Marketing Email",
                "color": "warning",
                "steps": [
                    "Do not reply to this email, as replying confirms your email address is active to spammers.",
                    "Avoid clicking 'Unsubscribe' links in suspicious or unknown emails, as they may lead to malicious landing pages.",
                    "Mark this email as Junk or Spam in your email client to train spam filters.",
                    "Do not purchase products or services advertised in unsolicited emails.",
                ],
            }
        else:
            return {
                "title": "✅ Legitimate Communication: General Best Practices",
                "color": "success",
                "steps": [
                    "This email appears legitimate and aligns with standard communications.",
                    "Always maintain baseline vigilance: verify sender identity before executing wire transfers or sharing sensitive personal data.",
                    "Ensure sender domain matches the expected organization.",
                ],
            }

    def predict(self, raw_input: str | bytes, model_name: str = "Best Model", attachment_names: list[str] | None = None) -> dict:
        """
        Complete end-to-end prediction pipeline.

        Args:
            raw_input: Raw email body string or .eml RFC 5322 text/bytes.
            model_name: Chosen model name.
            attachment_names: Optional list of attached filenames.

        Returns:
            dict containing prediction, probabilities, confidence, risk score,
                 top explainability features, highlighted text, red flags, and safety tips.
        """
        # 1. Header Analysis
        header_results = analyze_headers(raw_input)

        # 2. Extract content text
        if isinstance(raw_input, bytes):
            raw_str = raw_input.decode("utf-8", errors="replace")
        else:
            raw_str = raw_input

        # 3. Rule-based Feature Analysis (URLs, urgency, attachments)
        feature_results = extract_all_features(raw_str, attachment_names)

        # 4. Text Preprocessing
        cleaned_text = clean_text(raw_str)
        if not cleaned_text:
            cleaned_text = "empty email content"

        # 5. Model Selection & Probability Inference
        selected_model = self.models.get(model_name, self.models.get("Best Model"))
        if selected_model is None:
            selected_model = next(iter(self.models.values()))

        vec_input = self.vectorizer.transform([cleaned_text])
        prediction = selected_model.predict(vec_input)[0]

        # Probabilities
        if hasattr(selected_model, "predict_proba"):
            probs = selected_model.predict_proba(vec_input)[0]
            model_classes = list(getattr(selected_model, "classes_", CLASS_NAMES))
            prob_dict = {
                cls: round(float(probs[model_classes.index(cls)]), 4) for cls in CLASS_NAMES if cls in model_classes
            }
        else:
            # Fallback for models without predict_proba
            prob_dict = {cls: (1.0 if cls == prediction else 0.0) for cls in CLASS_NAMES}

        # Confidence Score
        confidence_score = round(max(prob_dict.values()) * 100.0, 2)
        phishing_prob = prob_dict.get("Phishing", 0.0)

        # 6. Explainability: Top 8 Words
        top_words = self._get_top_features_explanation(
            model=selected_model,
            model_name=model_name,
            clean_email=cleaned_text,
            predicted_class=prediction,
            top_k=8,
        )

        # 7. Highlighted Text
        highlighted_snippet = self._highlight_text(raw_str[:1200], top_words, prediction)

        # 8. Comprehensive Risk Score (0 - 100)
        risk_score, risk_level, risk_reasons = self.calculate_risk_score(
            phishing_prob=phishing_prob,
            features=feature_results,
            headers=header_results,
        )

        # 9. Dynamic Safety Tips
        safety_tips = self.get_safety_tips(prediction, risk_level)

        return {
            "prediction": prediction,
            "confidence_score": confidence_score,
            "probabilities": prob_dict,
            "model_used": model_name,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "top_words": top_words,
            "highlighted_text": highlighted_snippet,
            "clean_text": cleaned_text,
            "feature_analysis": feature_results,
            "header_analysis": header_results,
            "safety_tips": safety_tips,
        }


if __name__ == "__main__":
    predictor = PhishingPredictor()
    test_phish = """
    From: "PayPal Account Security" <service@paypal-alerts.com>
    Subject: Immediate Attention Required: Your Account Has Been Suspended!
    Date: Fri, 30 Oct 2026 21:00:00 +0000

    Dear Valued PayPal Customer,
    We noticed unauthorized logins to your PayPal account from an unrecognized IP address.
    To protect your funds, your account has been temporarily suspended.
    
    Please click here immediately to verify your identity and confirm your password:
    http://paypa1-security-verification.xyz/restore-account?id=99281
    
    Failure to verify your account within 24 hours will result in permanent account deletion.
    """
    print("\n--- Running Prediction Test on Phishing Email ---")
    result = predictor.predict(test_phish, model_name="Best Model")
    print(f"Prediction: {result['prediction']} (Confidence: {result['confidence_score']}%)")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Risk Score: {result['risk_score']}/100 [{result['risk_level']} Risk]")
    print("\nWhy this score:")
    for r in result["risk_reasons"]:
        print(f"  - {r}")
    print("\nTop 8 Influential Words:")
    for w in result["top_words"]:
        print(f"  {w['word']}: weight={w['weight']}")

