"""
AI-Based Email Phishing Detection & Threat Analysis Dashboard.
Streamlit Web Application - Light Theme Edition.

Features:
- Tab 1: Analyze (Real-time email analysis, .eml parser, explainability, risk gauge, PDF export)
- Tab 2: Batch Analysis (CSV file batch processing with progress bar and export)
- Tab 3: History (SQLite audit trail with label filtering and user feedback)
- Tab 4: Model Performance (Interactive benchmark metrics, confusion matrices, ROC)
- Tab 5: About (Architecture, NLP pipeline, dataset provenance, viva revision notes)
"""

from datetime import datetime
import email
from email import policy
from email.header import decode_header
import html
import json
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup import path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.features import extract_all_features
from src.headers import analyze_headers
from src.history import clear_history, get_history, get_history_stats, log_scan, record_feedback
from src.predict import PhishingPredictor
from src.report import generate_pdf_report

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Email Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Light Theme CSS
st.markdown(
    """
    
<style>
    /* Main Background & Typography */
    .stApp {
        background-color: #0E1117;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: #0E1117;
    }

    [data-testid="stSidebar"] {
        background-color: #1E293B;
    }

    .main .block-container {
        background-color: #0E1117;
    }

    /* Headings */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        font-size: 1.02rem;
        color: #CBD5E1;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }

    /* General Text */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    label, [data-testid="stWidgetLabel"] {
        color: #E2E8F0;
    }

    /* Result Cards */
    .card-legit {
        background-color: #123B2A;
        border: 1.5px solid #22C55E;
        padding: 1.25rem;
        border-radius: 10px;
        color: #BBF7D0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }

    .card-spam {
        background-color: #422F0B;
        border: 1.5px solid #F59E0B;
        padding: 1.25rem;
        border-radius: 10px;
        color: #FDE68A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }

    .card-phish {
        background-color: #451A1A;
        border: 1.5px solid #EF4444;
        padding: 1.25rem;
        border-radius: 10px;
        color: #FECACA;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }

    /* Security Finding Boxes */
    .flag-box-danger {
        background-color: #451A1A;
        border: 1px solid #EF4444;
        border-left: 4px solid #EF4444;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        color: #FECACA;
        font-size: 0.9rem;
    }

    .flag-box-warning {
        background-color: #422F0B;
        border: 1px solid #F59E0B;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        color: #FDE68A;
        font-size: 0.9rem;
    }

    .flag-box-safe {
        background-color: #123B2A;
        border: 1px solid #22C55E;
        border-left: 4px solid #22C55E;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        color: #BBF7D0;
        font-size: 0.9rem;
    }

    /* Email Highlight Container */
    .highlight-container {
        background: #1E293B;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.92rem;
        line-height: 1.65;
        color: #F1F5F9;
        max-height: 280px;
        overflow-y: auto;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.2);
    }

    .model-limit-note {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 8px;
        font-style: italic;
    }

    /* Input Fields and Text Areas */
    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1E293B;
        color: #F8FAFC;
        border-color: #475569;
    }

    /* File Upload */
    [data-testid="stFileUploader"] {
        background-color: #1E293B;
        border-radius: 8px;
        color: #F8FAFC;
    }

    /* Data Tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background-color: #1E293B;
        color: #F8FAFC;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px;
    }

    /* Navigation Tabs */
    button[data-baseweb="tab"] {
        color: #CBD5E1;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA;
    }
    /* Make Spam result card text clearly visible */

/* Bright and Readable Result Cards */

/* Legitimate - Green */
.card-legit {
    background-color: #123B2A !important;
    border: 2px solid #22C55E !important;
    color: #FFFFFF !important;
}

/* Spam - Amber */
.card-spam {
    background-color: #422F0B !important;
    border: 2px solid #FBBF24 !important;
    color: #FFFFFF !important;
}

/* Phishing - Red */
.card-phish {
    background-color: #451A1A !important;
    border: 2px solid #EF4444 !important;
    color: #FFFFFF !important;
}

/* Force all text inside the three cards to be bright */
.card-legit *,
.card-spam *,
.card-phish * {
    color: #FFFFFF !important;
}

</style>

    """,
    unsafe_allow_html=True,
)


def decode_str(s: str) -> str:
    """Safely decode email header string."""
    if not s:
        return ""
    try:
        decoded_fragments = decode_header(s)
        parts = []
        for text, encoding in decoded_fragments:
            if isinstance(text, bytes):
                enc = encoding or "utf-8"
                try:
                    parts.append(text.decode(enc, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    parts.append(text.decode("latin-1", errors="replace"))
            else:
                parts.append(str(text))
        return " ".join(parts).strip()
    except Exception:
        return str(s).strip()


def parse_eml_payload(file_bytes: bytes) -> dict:
    """Parse raw .eml RFC 5322 bytes into structured headers, body, and attachments."""
    try:
        msg = email.message_from_bytes(file_bytes, policy=policy.compat32)
    except Exception:
        msg = email.message_from_bytes(file_bytes)

    subject = decode_str(msg.get("subject", "No Subject"))
    from_val = decode_str(msg.get("from", "Unknown Sender"))
    to_val = decode_str(msg.get("to", ""))
    date_val = decode_str(msg.get("date", ""))

    attachments = []
    plain_parts = []
    html_parts = []

    try:
        if msg.is_multipart():
            for part in msg.walk():
                fn = part.get_filename()
                cd = str(part.get("Content-Disposition", ""))
                if fn or "attachment" in cd:
                    if fn:
                        attachments.append(decode_str(fn))
                    continue

                ctype = part.get_content_type()
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                charset = part.get_content_charset() or "utf-8"
                try:
                    text_content = payload.decode(charset, errors="replace")
                except Exception:
                    text_content = payload.decode("latin-1", errors="replace")

                if ctype == "text/plain":
                    plain_parts.append(text_content)
                elif ctype == "text/html":
                    html_parts.append(text_content)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    text_content = payload.decode(charset, errors="replace")
                except Exception:
                    text_content = payload.decode("latin-1", errors="replace")

                if msg.get_content_type() == "text/html":
                    html_parts.append(text_content)
                else:
                    plain_parts.append(text_content)
    except Exception:
        pass

    # Extract clean text from HTML if plain text missing
    clean_html_text = ""
    if html_parts:
        combined_html = "\n".join(html_parts)
        try:
            soup = BeautifulSoup(combined_html, "html.parser")
            for s in soup(["script", "style"]):
                s.decompose()
            clean_html_text = soup.get_text(separator="\n")
            clean_html_text = re.sub(r"\n\s*\n+", "\n\n", clean_html_text).strip()
        except Exception:
            clean_html_text = re.sub(r"<[^>]+>", " ", combined_html)

    body_text = "\n\n".join(plain_parts).strip()
    if not body_text and clean_html_text:
        body_text = clean_html_text

    raw_str = file_bytes.decode("utf-8", errors="replace")

    return {
        "subject": subject,
        "from": from_val,
        "to": to_val,
        "date": date_val,
        "attachments": attachments,
        "body_text": body_text,
        "raw_str": raw_str,
        "is_html_only": bool(not plain_parts and html_parts),
    }


@st.cache_resource
def load_predictor():
    """Cache the loaded ML models and TF-IDF vectorizer."""
    return PhishingPredictor(models_dir=BASE_DIR / "models")


predictor = load_predictor()

# ==========================================
# SIDEBAR CONTROLS & SETTINGS
# ==========================================
st.sidebar.markdown("### 🛡️ Security Engine")

available_models = predictor.get_available_models()
selected_model = st.sidebar.selectbox(
    "Active Classifier Model",
    options=available_models,
    index=0,
    help="Switch between the top-performing calibrated Linear SVM, baseline Logistic Regression, Naive Bayes, or Random Forest.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Sensitivity Calibration")
sensitivity = st.sidebar.select_slider(
    "Detection Sensitivity",
    options=["Lenient", "Normal", "Strict"],
    value="Normal",
    help="Adjusts risk-score thresholds. Strict mode lowers the threshold for High-Risk flags.",
)

if sensitivity == "Strict":
    high_threshold = 55.0
    med_threshold = 25.0
elif sensitivity == "Lenient":
    high_threshold = 80.0
    med_threshold = 45.0
else:
    high_threshold = 70.0
    med_threshold = 35.0

st.sidebar.caption(
    f"Thresholds: **High** ≥ {high_threshold:.0f} | **Medium** ≥ {med_threshold:.0f}"
)

# Display Model Accuracy Badge in Sidebar
metrics = predictor.metrics
if metrics and "models" in metrics and selected_model in metrics["models"]:
    m_info = metrics["models"][selected_model]
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Benchmark ({selected_model})**")
    col_sb1, col_sb2 = st.sidebar.columns(2)
    col_sb1.metric("Accuracy", f"{m_info['accuracy']*100:.1f}%")
    col_sb2.metric("Macro-F1", f"{m_info['macro_f1']:.3f}")

# ==========================================
# SAMPLE EMAILS (Realistic & Fictional)
# ==========================================
SAMPLE_LEGITIMATE = """From: aris.thorne@university-campus.edu
To: biology-students@university-campus.edu
Subject: Biology 301: Assignment 2 Feedback & Office Hours
Date: Fri, 30 Oct 2026 14:00:00 +0000
Message-ID: <bio301-fall2026@university-campus.edu>

Dear Students,

I have posted grades and comments for Assignment 2 on the course portal. If you would like to discuss your lab report or review the exam prep questions, my office hours are Tuesdays and Thursdays from 2 to 4 PM in Hall 410.

Please bring a copy of your experimental notes if you have specific questions about data analysis.

Best regards,
Dr. Aris Thorne
Department of Biological Sciences
University Science Campus
"""

SAMPLE_SPAM = """From: deals@newsletter-shopping-savings.com
To: user@example.com
Subject: Exclusive Online Clearance: Save 70% Today Only!
Date: Fri, 30 Oct 2026 10:15:00 +0000

Dear Valued Customer,

Big savings on vitamins, wellness supplements, and everyday health essentials!
Lowest prices guaranteed for the next 48 hours only.

Visit our online warehouse clearance to claim your exclusive discount coupon code.
Fast international shipping and 100% customer satisfaction guarantee.

Click here to view our digital clearance catalog: http://warehouse-bargain-clearance-outlet.com/shop

To unsubscribe from future offers, click unsubscribe here.
"""

SAMPLE_PHISHING = """From: "PayPal Account Security" <alert-update@paypa1-security-verification.xyz>
Reply-To: security-drop@fraudster-box.ru
Return-Path: bounce@marketing-blast.xyz
To: victim@example.com
Subject: URGENT: Your PayPal Account Has Been Suspended!
Date: Fri, 30 Oct 2026 21:00:00 +0000
Authentication-Results: mx.google.com; spf=fail; dkim=fail; dmarc=fail

Dear Customer,

We detected multiple unauthorized login attempts to your PayPal account from IP address 185.220.101.5 (Kyiv, Ukraine).
For your protection, we have temporarily suspended your account.

You must verify your identity immediately to restore account access:
http://192.168.1.100/paypal-login?session=98214&verify=account

If you fail to confirm your credentials within 24 hours, your account and all associated funds will be permanently deactivated.

Please also inspect the attached account statement: invoice_98214.pdf.exe

Thank you,
PayPal Security & Fraud Prevention Operations
"""

# ==========================================
# MAIN APPLICATION TABS
# ==========================================
tab_analyze, tab_batch, tab_history, tab_metrics, tab_about = st.tabs(
    ["🔍 Analyze Email", "📁 Batch Analysis", "📜 Scan History", "📊 Model Performance", "ℹ️ About & Viva"]
)

# ----------------------------------------------------
# TAB 1: ANALYZE EMAIL
# ----------------------------------------------------
with tab_analyze:
    st.markdown('<div class="main-header">AI Email Threat & Phishing Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Paste email text, inspect headers, or upload an RFC 5322 .eml file to assess legitimacy, spam, and phishing risks.</div>',
        unsafe_allow_html=True,
    )

    # Sample Buttons
    col_s1, col_s2, col_s3, col_s4 = st.columns([1, 1, 1, 2])
    if "email_input_content" not in st.session_state:
        st.session_state["email_input_content"] = ""

    if col_s1.button("📄 Load Legitimate Sample", use_container_width=True):
        st.session_state["email_input_content"] = SAMPLE_LEGITIMATE
    if col_s2.button("📢 Load Spam Sample", use_container_width=True):
        st.session_state["email_input_content"] = SAMPLE_SPAM
    if col_s3.button("🚨 Load Phishing Sample", use_container_width=True):
        st.session_state["email_input_content"] = SAMPLE_PHISHING

    # File Uploader (.eml)
    uploaded_eml = st.file_uploader(
        "Upload an RFC 5322 .eml file",
        type=["eml", "txt"],
        help="Upload an email file exported from Outlook, Apple Mail, Thunderbird, or Gmail.",
    )

    parsed_eml = None
    if uploaded_eml is not None:
        try:
            file_bytes = uploaded_eml.getvalue()
            parsed_eml = parse_eml_payload(file_bytes)
            st.session_state["parsed_eml"] = parsed_eml
            st.session_state["uploaded_eml_name"] = uploaded_eml.name

            att_info = f"({', '.join(parsed_eml['attachments'])})" if parsed_eml["attachments"] else "(None)"
            st.markdown(
                f"""
                <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; color: #1E40AF;">
                    <strong>📬 Uploaded .EML File:</strong> {html.escape(uploaded_eml.name)} ({len(file_bytes)/1024:.1f} KB)<br>
                    <span style="font-size: 0.88rem; color: #1E3A8A;">
                    <strong>Subject:</strong> {html.escape(parsed_eml['subject'])} | 
                    <strong>From:</strong> {html.escape(parsed_eml['from'])} | 
                    <strong>Attachments:</strong> {len(parsed_eml['attachments'])} {html.escape(att_info)}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error parsing .eml file: {e}")

    email_input = st.text_area(
        "Email Content (Headers & Body)",
        value=st.session_state.get("email_input_content", ""),
        height=190,
        placeholder="Paste full email text or RFC 5322 email headers here (or click 'Analyze Email' to scan the uploaded .eml file above)...",
    )

    analyze_clicked = st.button("🚀 Analyze Email", type="primary", use_container_width=True)

    # Determine input source
    has_typed_input = bool(email_input and email_input.strip())
    has_uploaded_eml = bool(parsed_eml is not None)

    if analyze_clicked or ("last_scan_result" in st.session_state and (has_typed_input or has_uploaded_eml)):
        raw_text = ""
        attachment_names = None

        if has_typed_input:
            raw_text = email_input.strip()
            if has_uploaded_eml and parsed_eml:
                attachment_names = parsed_eml.get("attachments")
        elif has_uploaded_eml and parsed_eml:
            raw_text = parsed_eml["raw_str"]
            attachment_names = parsed_eml.get("attachments")

        # Input validation
        if not raw_text:
            st.warning("⚠️ Please paste email text or upload a .eml file before analyzing.")
        elif len(raw_text) < 20:
            st.warning("⚠️ The provided email snippet is too short (< 20 characters) for reliable classification.")
        else:
            if len(raw_text) > 15000:
                st.info("ℹ️ Note: Input is very long (> 15,000 characters). Analysis is proceeding on the complete text.")

            ascii_chars = sum(1 for c in raw_text if ord(c) < 128)
            if len(raw_text) > 0 and (ascii_chars / len(raw_text)) < 0.60:
                st.warning("⚠️ Non-English characters detected. The models are trained on English corpora; non-English text may degrade classification accuracy.")

            # Perform prediction
            with st.spinner("Analyzing email structure, headers, and linguistic features..."):
                scan_res = predictor.predict(raw_text, model_name=selected_model, attachment_names=attachment_names)
                st.session_state["last_scan_result"] = scan_res
                st.session_state["last_scanned_text"] = raw_text

                # Apply sensitivity thresholds
                r_score = scan_res["risk_score"]
                if r_score >= high_threshold:
                    eff_risk_level = "High"
                elif r_score >= med_threshold:
                    eff_risk_level = "Medium"
                else:
                    eff_risk_level = "Low"
                scan_res["risk_level"] = eff_risk_level

                # Log scan in history
                if "last_logged_id" not in st.session_state or st.session_state.get("last_logged_text") != raw_text:
                    rec_id = log_scan(
                        email_snippet=raw_text[:150],
                        prediction=scan_res["prediction"],
                        confidence=scan_res["confidence_score"],
                        risk_score=scan_res["risk_score"],
                        risk_level=eff_risk_level,
                        model_used=selected_model,
                    )
                    st.session_state["last_logged_id"] = rec_id
                    st.session_state["last_logged_text"] = raw_text

            # -----------------------------------------
            # DISPLAY RESULTS
            # -----------------------------------------
            st.markdown("---")
            pred = scan_res["prediction"]
            conf = scan_res["confidence_score"]
            probs = scan_res["probabilities"]

            col_res1, col_res2, col_res3 = st.columns([1.4, 1.3, 1.3])

            with col_res1:
                if pred == "Phishing":
                    st.markdown(
                        f"""
                        <div class="card-phish">
                            <h3 style="margin-top: 0; color: #7F1D1D;">🚨 Phishing Attack Detected</h3>
                            <p style="margin: 0; font-size: 1.05rem; color: #991B1B;"><strong>Confidence:</strong> {conf:.1f}%</p>
                            <p style="margin: 0; font-size: 0.88rem; color: #B91C1C;">Model: {selected_model}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif pred == "Spam":
                    st.markdown(
                        f"""
                        <div class="card-spam">
                            <h3 style="margin-top: 0; color: #78350F;">⚠️ Unsolicited Spam Email</h3>
                            <p style="margin: 0; font-size: 1.05rem; color: #92400E;"><strong>Confidence:</strong> {conf:.1f}%</p>
                            <p style="margin: 0; font-size: 0.88rem; color: #B45309;">Model: {selected_model}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="card-legit">
                            <h3 style="margin-top: 0; color: #14532D;">✅ Legitimate Email</h3>
                            <p style="margin: 0; font-size: 1.05rem; color: #166534;"><strong>Confidence:</strong> {conf:.1f}%</p>
                            <p style="margin: 0; font-size: 0.88rem; color: #15803D;">Model: {selected_model}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Mandatory Model Limitation Note
                st.markdown(
                    '<p class="model-limit-note">Model trained on public email corpora from 2003-2005; verify with other signals for modern phishing.</p>',
                    unsafe_allow_html=True,
                )


            with col_res2:
                # Plotly Probability Bar Chart (Dark Theme)
                fig_prob = go.Figure(
                    go.Bar(
                        x=[
                            probs.get("Legitimate", 0) * 100,
                            probs.get("Spam", 0) * 100,
                            probs.get("Phishing", 0) * 100
                        ],
                        y=["Legitimate", "Spam", "Phishing"],
                        orientation="h",
                        marker=dict(color=["#10B981", "#F59E0B", "#EF4444"]),
                        text=[
                            f"{probs.get(c, 0)*100:.1f}%"
                            for c in ["Legitimate", "Spam", "Phishing"]
                        ],
                        textposition="auto",
                        textfont=dict(color="#FFFFFF", size=12),
                    )
                )

                fig_prob.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    title={
                        "text": "Class Probabilities",
                        "font": {"size": 15, "color": "#F8FAFC"}
                    },
                    xaxis={
                        "title": {
                            "text": "Probability (%)",
                            "font": {"color": "#F8FAFC"}
                        },
                        "range": [0, 100],
                        "color": "#F8FAFC",
                        "gridcolor": "#334155",
                        "zerolinecolor": "#475569"
                    },
                    yaxis={
                        "color": "#F8FAFC",
                        "gridcolor": "#334155"
                    },
                    height=165,
                    margin=dict(l=10, r=10, t=30, b=20),
                    font={"color": "#F8FAFC"}
                )

                st.plotly_chart(fig_prob, use_container_width=True)

            with col_res3:
                # Plotly Gauge Chart for Risk Score (Dark Theme)
                gauge_color = (
                    "#EF4444" if eff_risk_level == "High"
                    else ("#F59E0B" if eff_risk_level == "Medium" else "#10B981")
                )

                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=scan_res["risk_score"],
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={
                            "text": f"Risk Score [{eff_risk_level} Risk]",
                            "font": {"size": 15, "color": "#F8FAFC"}
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickwidth": 1,
                                "tickcolor": "#F8FAFC",
                                "tickfont": {"color": "#F8FAFC"}
                            },
                            "bar": {"color": gauge_color},
                            "bgcolor": "#1E293B",
                            "borderwidth": 1,
                            "bordercolor": "#475569",
                            "steps": [
                                {"range": [0, med_threshold], "color": "#14532D"},
                                {"range": [med_threshold, high_threshold], "color": "#78350F"},
                                {"range": [high_threshold, 100], "color": "#7F1D1D"},
                            ],
                            "threshold": {
                                "line": {"color": "#F8FAFC", "width": 3},
                                "thickness": 0.75,
                                "value": scan_res["risk_score"],
                            },
                        },
                        number={
                            "font": {"color": "#F8FAFC", "size": 28}
                        },
                    )
                )

                fig_gauge.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=165,
                    margin=dict(l=20, r=20, t=30, b=10),
                    font={"color": "#F8FAFC"}
                )

                st.plotly_chart(fig_gauge, use_container_width=True)


            # -----------------------------------------
            # EXPLAINABILITY: READABLE PARAGRAPH WITH INLINE HIGHLIGHTS
            # -----------------------------------------
            st.markdown("### 🧠 Explainability & Evidence")
            col_exp1, col_exp2 = st.columns([1, 1.25])

            with col_exp1:
                st.markdown("**Top Influential Words Toward Prediction:**")
                top_words = scan_res.get("top_words", [])
                if top_words:
                    word_df = pd.DataFrame(top_words)
                    word_df.columns = ["Keyword / N-gram", "Attribution Weight", "TF-IDF Weight"]
                    st.dataframe(word_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No specific vocabulary features exerted strong positive attribution.")

            with col_exp2:
                st.markdown("**Email Content (Keywords Highlighted Inline):**")
                st.markdown(
                    f"""
                    <div class="highlight-container">
                        {scan_res['highlighted_text']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -----------------------------------------
            # RED FLAGS & HEURISTICS BREAKDOWN
            # -----------------------------------------
            st.markdown("### 🔍 Security Analysis Breakdown")
            col_f1, col_f2, col_f3 = st.columns(3)

            feat = scan_res["feature_analysis"]
            with col_f1:
                st.markdown("**🔗 URL & Link Inspection**")
                urls = feat["url_analysis"]
                st.write(f"Total links: **{urls['total_urls_found']}** | Suspicious: **{urls['suspicious_url_count']}**")
                if urls["red_flags"]:
                    for rf in urls["red_flags"]:
                        st.markdown(f'<div class="flag-box-danger">⚠️ {html.escape(rf)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="flag-box-safe">✓ No suspicious link anomalies detected.</div>', unsafe_allow_html=True)

            with col_f2:
                st.markdown("**⚡ Urgency & Threat Triggers**")
                urgency = feat["urgency_analysis"]
                st.write(f"Total triggers: **{urgency['total_triggers']}** | Categories: **{urgency['categories_triggered']}**")
                if urgency["breakdown"]:
                    for cat, words in urgency["breakdown"].items():
                        st.markdown(
                            f'<div class="flag-box-warning"><strong>{html.escape(cat)}:</strong> {html.escape(", ".join(words[:3]))}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown('<div class="flag-box-safe">✓ No coercive or urgency language detected.</div>', unsafe_allow_html=True)

            with col_f3:
                st.markdown("**📎 Attachments & Headers**")
                att = feat["attachment_analysis"]
                st.write(f"Suspicious attachments: **{att['suspicious_attachments_count']}**")
                if att["red_flags"]:
                    for af in att["red_flags"]:
                        st.markdown(f'<div class="flag-box-danger">🚨 {html.escape(af)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="flag-box-safe">✓ No dangerous attachment extensions found.</div>', unsafe_allow_html=True)

                hdrs = scan_res["header_analysis"]
                if hdrs.get("headers_present"):
                    if hdrs["red_flags"]:
                        for hf in hdrs["red_flags"]:
                            st.markdown(f'<div class="flag-box-danger">🛡️ {html.escape(hf)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="flag-box-safe">✓ Email headers verified authentic.</div>', unsafe_allow_html=True)
                else:
                    st.caption("• Headers: *Skipped (Pasted email body)*")

            # -----------------------------------------
            # WHY THIS SCORE & SAFETY TIPS
            # -----------------------------------------
            st.markdown("### 📋 Risk Rationale & Actionable Safety Tips")
            col_r1, col_r2 = st.columns([1, 1])

            with col_r1:
                st.markdown("**Why this score?**")
                for reason in scan_res["risk_reasons"]:
                    st.markdown(f"- {reason}")

            with col_r2:
                tips = scan_res["safety_tips"]
                st.markdown(f"**{tips['title']}**")
                for step in tips["steps"]:
                    st.markdown(f"- {step}")

            # -----------------------------------------
            # PDF EXPORT & FEEDBACK
            # -----------------------------------------
            st.markdown("---")
            col_act1, col_act2, col_act3 = st.columns([1.5, 1, 1])

            with col_act1:
                try:
                    pdf_data = generate_pdf_report(scan_res, raw_text)
                    st.download_button(
                        label="📥 Download PDF Threat Report",
                        data=pdf_data,
                        file_name=f"Threat_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF: {e}")

            with col_act2:
                if st.button("👍 Prediction is Correct", use_container_width=True):
                    if "last_logged_id" in st.session_state:
                        record_feedback(st.session_state["last_logged_id"], is_correct=True)
                        st.success("Thank you for your feedback! Stored in audit log.")

            with col_act3:
                if st.button("👎 Prediction is Incorrect", use_container_width=True):
                    if "last_logged_id" in st.session_state:
                        record_feedback(st.session_state["last_logged_id"], is_correct=False)
                        st.warning("Feedback logged! This example will be flagged for review.")

# ----------------------------------------------------
# TAB 2: BATCH ANALYSIS
# ----------------------------------------------------
with tab_batch:
    st.markdown('<div class="main-header">Batch Email Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a CSV file of emails to classify in bulk with progress tracking and export.</div>',
        unsafe_allow_html=True,
    )

    batch_file = st.file_uploader("Upload CSV file for batch scanning", type=["csv"], key="batch_uploader")

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.write(f"Uploaded file contains **{len(batch_df):,}** rows and **{len(batch_df.columns)}** columns.")
            st.dataframe(batch_df.head(3), use_container_width=True)

            text_col = st.selectbox(
                "Select column containing email text:",
                options=batch_df.columns.tolist(),
                index=0,
            )

            row_cap = st.slider("Maximum rows to classify:", min_value=10, max_value=500, value=100, step=10)

            if st.button("⚡ Start Batch Classification", type="primary"):
                df_to_process = batch_df.head(row_cap).copy()
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []
                total = len(df_to_process)

                for idx, row in df_to_process.iterrows():
                    text_val = str(row[text_col])
                    res = predictor.predict(text_val, model_name=selected_model)
                    results.append({
                        "row_index": idx,
                        "email_snippet": text_val[:120].replace("\n", " "),
                        "predicted_label": res["prediction"],
                        "confidence_percent": res["confidence_score"],
                        "risk_score": res["risk_score"],
                        "risk_level": res["risk_level"],
                        "p_phishing": res["probabilities"].get("Phishing", 0.0),
                    })
                    progress_bar.progress((len(results)) / total)
                    status_text.text(f"Processed {len(results)} / {total} emails...")

                res_df = pd.DataFrame(results)
                status_text.success(f"Batch analysis complete for {len(res_df)} emails!")

                col_b1, col_b2 = st.columns([1, 1.5])
                with col_b1:
                    st.markdown("**Label Distribution:**")
                    counts = res_df["predicted_label"].value_counts().reset_index()
                    counts.columns = ["Class", "Count"]
                    fig_dist = px.pie(
                        counts,
                        names="Class",
                        values="Count",
                        color="Class",
                        color_discrete_map={"Legitimate": "#10B981", "Spam": "#F59E0B", "Phishing": "#EF4444"},
                        hole=0.4,
                    )
                    fig_dist.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=260,
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col_b2:
                    st.markdown("**Summary Metrics:**")
                    c_legit = (res_df["predicted_label"] == "Legitimate").sum()
                    c_spam = (res_df["predicted_label"] == "Spam").sum()
                    c_phish = (res_df["predicted_label"] == "Phishing").sum()
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Legitimate", c_legit)
                    col_m2.metric("Spam", c_spam)
                    col_m3.metric("Phishing", c_phish)
                    avg_risk = res_df["risk_score"].mean()
                    st.info(f"Average Batch Risk Score: **{avg_risk:.1f} / 100**")

                st.dataframe(res_df, use_container_width=True)

                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=csv_bytes,
                    file_name=f"batch_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ----------------------------------------------------
# TAB 3: SCAN HISTORY
# ----------------------------------------------------
with tab_history:
    st.markdown('<div class="main-header">Scan History & Audit Log</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Persistent audit log of email scans stored in SQLite with user feedback tracking.</div>',
        unsafe_allow_html=True,
    )

    stats = get_history_stats()
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("Total Scans", stats.get("total_scans", 0))
    col_h2.metric("Legitimate", stats.get("counts_by_label", {}).get("Legitimate", 0))
    col_h3.metric("Spam", stats.get("counts_by_label", {}).get("Spam", 0))
    col_h4.metric("Phishing", stats.get("counts_by_label", {}).get("Phishing", 0))

    st.markdown("---")
    col_filt, col_clr = st.columns([3, 1])

    with col_filt:
        filter_label = st.selectbox(
            "Filter History by Predicted Class:",
            options=["All", "Legitimate", "Spam", "Phishing"],
            index=0,
        )

    with col_clr:
        confirm_clear = st.checkbox("Confirm Clear History")
        if st.button("🗑️ Clear All History", type="secondary", disabled=not confirm_clear):
            clear_history()
            st.success("Scan history database cleared!")
            st.rerun()

    history_df = get_history(limit=200, label_filter=filter_label)

    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No scan records found in the database. Scan an email in the 'Analyze' tab to populate history.")

# ----------------------------------------------------
# TAB 4: MODEL PERFORMANCE
# ----------------------------------------------------
with tab_metrics:
    st.markdown('<div class="main-header">Model Performance & Benchmarking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluation metrics from 5-fold cross-validation and held-out test evaluation.</div>',
        unsafe_allow_html=True,
    )

    metrics_data = predictor.metrics
    if metrics_data and "models" in metrics_data:
        table_rows = []
        for m_name, m_res in metrics_data["models"].items():
            table_rows.append({
                "Model Name": m_name,
                "Accuracy (%)": round(m_res["accuracy"] * 100, 2),
                "Macro-F1": round(m_res["macro_f1"], 4),
                "Phishing-F1": round(m_res["per_class"]["Phishing"]["f1_score"], 4),
                "Macro-AUC": round(m_res["macro_auc"], 4),
                "5-Fold CV F1": round(m_res["cv_macro_f1"], 4),
                "Best Hyperparameters": str(m_res["best_params"]),
            })

        comp_df = pd.DataFrame(table_rows)
        st.markdown("### 🏆 Model Comparison Benchmark")
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        reports_dir = BASE_DIR / "reports"
        st.markdown("### 📈 Visual Benchmark Reports")

        col_p1, col_p2 = st.columns(2)
        cm_path = reports_dir / "confusion_matrices_all.png"
        if cm_path.exists():
            col_p1.image(str(cm_path), caption="Confusion Matrices Across All Models (Held-out Test Set)")

        roc_path = reports_dir / "roc_comparison.png"
        if roc_path.exists():
            col_p2.image(str(roc_path), caption="Phishing Class ROC Curves (Sensitivity vs False Alarms)")

        col_p3, col_p4 = st.columns(2)
        bar_path = reports_dir / "model_comparison.png"
        if bar_path.exists():
            col_p3.image(str(bar_path), caption="Multi-Metric Comparative Performance Bar Chart")

        top_words_path = reports_dir / "eda_top_words.png"
        if top_words_path.exists():
            col_p4.image(str(top_words_path), caption="Top Keywords Identified in Dataset Corpora")

    else:
        st.warning("Metrics file 'models/metrics.json' not found. Please run 'python src/train.py' first.")

