"""
Preprocessing Module for AI-Based Email Phishing Detection.
Provides a unified, explainable clean_text() pipeline for both
training and real-time inference.

Pipeline steps:
1. HTML tag removal (BeautifulSoup & regex fallback)
2. Lowercasing & unicode normalization
3. Entity tokenization:
   - URLs -> 'urltoken'
   - Emails -> 'emailtoken'
   - Currency -> 'currencytoken'
   - Numbers -> 'numbertoken'
4. Punctuation removal
5. Noise-token filtering:
   - Tokens > 20 characters (base64 blobs, long IDs)
   - Pure hex strings (2-8 chars like '3d', '7f', 'ff')
   - Tokens with no vowels and <= 4 chars (header noise like 'zw', 'nc', 'gb')
   - Tokens that are entirely non-alphabetic
6. Stopword removal (NLTK English corpus / bundled set)
7. Lemmatization (NLTK WordNetLemmatizer)

NOTE: clean_text() is designed to receive BODY TEXT ONLY (Subject + body).
      Headers must be stripped before calling this function.
      Use extract_email_body() in predict.py for that separation.
"""

import html
from pathlib import Path
import re
import unicodedata
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure project-local NLTK data directory
LOCAL_NLTK_DIR = Path(__file__).resolve().parent.parent / "data" / "nltk_data"
if LOCAL_NLTK_DIR.exists():
    nltk.data.path.insert(0, str(LOCAL_NLTK_DIR))

# Standard English Stopwords fallback
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = FALLBACK_STOPWORDS

try:
    LEMMATIZER = WordNetLemmatizer()
    _ = LEMMATIZER.lemmatize("running")
except Exception:
    # If WordNet not accessible, provide lightweight suffix lemmatizer
    class SimpleLemmatizer:
        def lemmatize(self, word: str, pos: str = "n") -> str:
            if word.endswith("ies") and len(word) > 4:
                return word[:-3] + "y"
            if word.endswith("es") and len(word) > 3:
                return word[:-2]
            if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
                return word[:-1]
            if word.endswith("ing") and len(word) > 4:
                return word[:-3]
            if word.endswith("ed") and len(word) > 3:
                return word[:-2]
            return word
    LEMMATIZER = SimpleLemmatizer()

# Precompiled regex patterns for speed and consistency
URL_PATTERN = re.compile(
    r"(?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CURRENCY_PATTERN = re.compile(r"[\$\£\€\¥]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:dollars?|usd|eur|gbp)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
NON_WORD_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean and normalize email text for TF-IDF feature extraction.
    Used identically during model training and real-time prediction.

    Args:
        text (str): Raw email text (can include HTML, headers, or plain body).

    Returns:
        str: Lemmatized, tokenized, and stopword-free lowercase string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (&amp;, &lt;, etc.)
    text = html.unescape(text)

    # 2. Strip HTML tags if present
    if "<" in text and ">" in text:
        try:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ")
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)

    # 3. Unicode normalization (NFKD)
    text = unicodedata.normalize("NFKD", text)

    # 4. Lowercase
    text = text.lower()

    # 5. Replace URLs with 'urltoken'
    text = URL_PATTERN.sub(" urltoken ", text)

    # 6. Replace Email addresses with 'emailtoken'
    text = EMAIL_PATTERN.sub(" emailtoken ", text)

    # 7. Replace Currency mentions with 'currencytoken'
    text = CURRENCY_PATTERN.sub(" currencytoken ", text)

    # 8. Replace remaining numbers with 'numbertoken'
    text = NUMBER_PATTERN.sub(" numbertoken ", text)

    # 9. Remove punctuation and non-alphabetic characters
    text = NON_WORD_PATTERN.sub(" ", text)

    # 10. Split tokens, remove stopwords, and lemmatize
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        # Keep our special marker tokens intact
        if token in {"urltoken", "emailtoken", "currencytoken", "numbertoken"}:
            cleaned_tokens.append(token)
            continue

        # Skip stopwords and single character noise
        if token not in STOPWORDS and len(token) > 1:
            lemma = LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


if __name__ == "__main__":
    sample_email = """
    Subject: URGENT: Verify Your PayPal Account!
    Dear Customer,
    We detected suspicious logins from IP 192.168.1.50. Please click <a href="http://paypal-security-update.com/login?id=9928">http://paypal-security-update.com</a> immediately to confirm your identity or your account will be suspended within 24 hours.
    Contact us at support@paypal-security.org or pay $50.00 reinstatement fee.
    """
    print("--- RAW EMAIL ---")
    print(sample_email)
    print("\n--- PREPROCESSED TEXT ---")
    cleaned = clean_text(sample_email)
    print(cleaned)
