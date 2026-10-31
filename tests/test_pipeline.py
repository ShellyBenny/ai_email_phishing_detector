"""
Comprehensive Pytest Test Suite for AI-Based Email Phishing Detection.
Tests:
- Preprocessing (HTML stripping, tokenization, lemmatization)
- Feature extraction (URL red flags, lookalikes, urgency lexicon, attachments)
- Email header analysis (mismatches, authentication verdicts, graceful fallback)
- Model prediction, probability calibration, risk scoring, and explainability
"""

from pathlib import Path
import pytest
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    analyze_attachments,
    analyze_urgency_and_threats,
    analyze_urls,
    check_brand_lookalike,
    extract_all_features,
    levenshtein_distance,
)
from src.headers import analyze_headers, extract_domain
from src.predict import PhishingPredictor
from src.preprocess import clean_text


class TestPreprocessing:
    """Tests for src/preprocess.py."""

    def test_clean_text_basic(self):
        sample = "Hello World! This is a simple test email."
        cleaned = clean_text(sample)
        assert "hello" in cleaned
        assert "test" in cleaned
        assert "!" not in cleaned

    def test_html_stripping(self):
        sample = "<html><body><h1>Security Notice</h1><p>Please click here.</p></body></html>"
        cleaned = clean_text(sample)
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "security" in cleaned
        assert "notice" in cleaned

    def test_url_token_replacement(self):
        sample = "Please verify your account at https://secure-login.com/auth today."
        cleaned = clean_text(sample)
        assert "urltoken" in cleaned
        assert "https://" not in cleaned

    def test_email_token_replacement(self):
        sample = "Contact support at billing-dept@company.org for assistance."
        cleaned = clean_text(sample)
        assert "emailtoken" in cleaned
        assert "@" not in cleaned

    def test_currency_and_number_replacement(self):
        sample = "Pay $150.00 within 24 hours to avoid 5 penalty fees."
        cleaned = clean_text(sample)
        assert "currencytoken" in cleaned
        assert "numbertoken" in cleaned


class TestFeatureExtraction:
    """Tests for src/features.py."""

    def test_levenshtein_distance(self):
        assert levenshtein_distance("paypal", "paypal") == 0
        assert levenshtein_distance("paypal", "paypa1") == 1
        assert levenshtein_distance("amazon", "amaz0n") == 1

    def test_lookalike_domain_detection(self):
        hit1 = check_brand_lookalike("paypa1-security.com")
        assert hit1 is not None
        assert hit1["brand"] == "paypal"

        hit2 = check_brand_lookalike("amaz0n-prime-reward.xyz")
        assert hit2 is not None
        assert hit2["brand"] == "amazon"

        # Legitimate domain should not trigger
        assert check_brand_lookalike("paypal.com") is None
        assert check_brand_lookalike("apple.com") is None

    def test_url_red_flags_ip_and_tld(self):
        raw = "Visit http://192.168.1.1/login or http://free-prize.xyz/claim"
        res = analyze_urls(raw)
        assert res["total_urls_found"] >= 2
        flags = " ".join(res["red_flags"])
        assert "raw IP" in flags
        assert ".xyz" in flags

    def test_html_link_text_mismatch(self):
        html_email = '<a href="http://evil-attacker.ru/steal">https://paypal.com/signin</a>'
        res = analyze_urls(html_email)
        flags = " ".join(res["red_flags"])
        assert "mismatch" in flags.lower()

    def test_urgency_and_threat_detection(self):
        text = "URGENT: Your account has been suspended! You must verify your account within 24 hours."
        res = analyze_urgency_and_threats(text)
        assert res["total_triggers"] >= 3
        assert "Urgency" in res["breakdown"]
        assert "Threat / Fear" in res["breakdown"]
        assert "Credential Request" in res["breakdown"]

    def test_attachment_double_extension(self):
        text = "Please open the attached file: report.pdf.exe and statement.doc.vbs"
        res = analyze_attachments(text)
        assert res["suspicious_attachments_count"] >= 2
        flags = " ".join(res["red_flags"])
        assert "double extension" in flags.lower()


class TestHeaderAnalysis:
    """Tests for src/headers.py."""

    def test_pasted_body_graceful_skip(self):
        body = "Hi team, let's meet at 2 PM for sprint review. Thanks!"
        res = analyze_headers(body)
        assert res["headers_present"] is False
        assert res["header_risk_penalty"] == 0.0

    def test_from_reply_to_mismatch(self):
        eml = (
            "From: PayPal Support <support@paypal.com>\n"
            "Reply-To: attacker-drop@fraudster.net\n"
            "Subject: Important Update\n"
            "Date: Fri, 30 Oct 2026 12:00:00 +0000\n\n"
            "Please read this notice."
        )
        res = analyze_headers(eml)
        assert res["headers_present"] is True
        flags = " ".join(res["red_flags"])
        assert "Reply-To mismatch" in flags

    def test_display_name_spoofing(self):
        eml = (
            'From: "PayPal Security Team" <hacker@scam-domain.org>\n'
            "Subject: Security Warning\n"
            "Date: Fri, 30 Oct 2026 12:00:00 +0000\n\n"
            "Action required."
        )
        res = analyze_headers(eml)
        assert res["headers_present"] is True
        flags = " ".join(res["red_flags"])
        assert "Brand impersonation" in flags


class TestPredictionAndExplainability:
    """Tests for src/predict.py."""

    @pytest.fixture(scope="class")
    def predictor(self):
        return PhishingPredictor()

    def test_predictor_loaded_models(self, predictor):
        available = predictor.get_available_models()
        assert "Best Model" in available
        assert "Naive Bayes" in available
        assert "Logistic Regression" in available

    def test_phishing_prediction_and_risk_score(self, predictor):
        phish_email = """
        From: "PayPal Notice" <alert@paypa1-security.xyz>
        Subject: URGENT: Your PayPal Account Has Been Suspended!
        
        Dear customer,
        We detected unauthorized logins from an unknown IP address.
        Click here immediately to verify your account or your account will be terminated within 24 hours:
        http://192.168.1.105/login
        
        Attached: invoice.pdf.exe
        """
        result = predictor.predict(phish_email, model_name="Best Model")
        assert result["prediction"] == "Phishing"
        assert result["risk_score"] > 60.0
        assert result["risk_level"] in {"High", "Medium"}
        assert len(result["top_words"]) > 0
        assert "Action Required" in result["safety_tips"]["title"]

    def test_model_switching(self, predictor):
        sample = "Subject: Urgent Password Reset Required\nPlease click here to reset your password immediately."
        res_nb = predictor.predict(sample, model_name="Naive Bayes")
        res_lr = predictor.predict(sample, model_name="Logistic Regression")
        assert res_nb["model_used"] == "Naive Bayes"
        assert res_lr["model_used"] == "Logistic Regression"
        assert "probabilities" in res_nb
        assert "probabilities" in res_lr

