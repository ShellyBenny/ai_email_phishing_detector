"""
Dataset Loader and EDA Generator for AI-Based Email Phishing Detection.
Downloads and processes public datasets:
  - Legitimate (Ham): Apache SpamAssassin (easy_ham, easy_ham_2, hard_ham)
  - Spam: Apache SpamAssassin (spam, spam_2)
  - Phishing: Jose Nazario Phishing Corpus (phishing0.mbox, phishing1.mbox, phishing2.mbox)

Labels:
  - Legitimate
  - Spam
  - Phishing
"""

import email
from email import policy
from email.header import decode_header
import json
import mailbox
import os
from pathlib import Path
import re
import tarfile
import urllib.request
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_CSV = DATA_DIR / "emails.csv"

# Dataset sources
DATASET_SOURCES = {
    "Legitimate": [
        {
            "name": "SpamAssassin_easy_ham",
            "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
            "filename": "20030228_easy_ham.tar.bz2",
            "type": "tarbz2",
            "subfolder": "easy_ham",
        },
        {
            "name": "SpamAssassin_easy_ham_2",
            "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
            "filename": "20030228_easy_ham_2.tar.bz2",
            "type": "tarbz2",
            "subfolder": "easy_ham_2",
        },
        {
            "name": "SpamAssassin_hard_ham",
            "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
            "filename": "20030228_hard_ham.tar.bz2",
            "type": "tarbz2",
            "subfolder": "hard_ham",
        },
    ],
    "Spam": [
        {
            "name": "SpamAssassin_spam",
            "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
            "filename": "20030228_spam.tar.bz2",
            "type": "tarbz2",
            "subfolder": "spam",
        },
        {
            "name": "SpamAssassin_spam_2",
            "url": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
            "filename": "20050311_spam_2.tar.bz2",
            "type": "tarbz2",
            "subfolder": "spam_2",
        },
    ],
    "Phishing": [
        {
            "name": "Nazario_phishing0",
            "url": "https://monkey.org/~jose/phishing/phishing0.mbox",
            "filename": "phishing0.mbox",
            "type": "mbox",
        },
        {
            "name": "Nazario_phishing1",
            "url": "https://monkey.org/~jose/phishing/phishing1.mbox",
            "filename": "phishing1.mbox",
            "type": "mbox",
        },
        {
            "name": "Nazario_phishing2",
            "url": "https://monkey.org/~jose/phishing/phishing2.mbox",
            "filename": "phishing2.mbox",
            "type": "mbox",
        },
    ],
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download file from url to dest_path with user-agent header."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [Cache hit] {dest_path.name} already exists ({dest_path.stat().st_size} bytes).")
        return True

    print(f"  Downloading {url} -> {dest_path.name}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AI-Email-Detector-Research/1.0)"})
        with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
            chunk_size = 64 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"  Successfully downloaded {dest_path.name} ({dest_path.stat().st_size} bytes).")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


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


def extract_body_and_subject(msg) -> tuple[str, str]:
    """Extract subject and clean body text from email message."""
    subject = decode_str(msg.get("subject", ""))
    body_parts = []

    try:
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in content_disposition:
                    continue
                try:
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        text = payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        text = payload.decode("latin-1", errors="replace")

                    if content_type == "text/plain":
                        body_parts.append(text)
                    elif content_type == "text/html":
                        soup = BeautifulSoup(text, "html.parser")
                        body_parts.append(soup.get_text(separator=" "))
                except Exception:
                    continue
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    text = payload.decode("latin-1", errors="replace")
                if msg.get_content_type() == "text/html":
                    soup = BeautifulSoup(text, "html.parser")
                    body_parts.append(soup.get_text(separator=" "))
                else:
                    body_parts.append(text)
    except Exception:
        pass

    full_body = "\n".join(body_parts).strip()
    return subject, full_body


def parse_tarbz2(tar_path: Path, source_name: str, label: str) -> list[dict]:
    """Extract email messages from a tar.bz2 archive."""
    records = []
    print(f"  Parsing archive: {tar_path.name}...")
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.endswith(".cmds") or member.name.endswith("/"):
                    continue
                f = tar.extractfile(member)
                if not f:
                    continue
                raw_bytes = f.read()
                try:
                    msg = email.message_from_bytes(raw_bytes, policy=policy.compat32)
                    subject, body = extract_body_and_subject(msg)
                    full_text = f"Subject: {subject}\n\n{body}".strip() if subject else body.strip()
                    if full_text:
                        records.append({
                            "text": full_text,
                            "label": label,
                            "source": source_name,
                        })
                except Exception:
                    continue
    except Exception as e:
        print(f"  [ERROR] Parsing tar.bz2 failed: {e}")
    print(f"    Extracted {len(records)} messages from {source_name}.")
    return records


def parse_mbox(mbox_path: Path, source_name: str, label: str) -> list[dict]:
    """Extract email messages from an mbox file."""
    records = []
    print(f"  Parsing mbox: {mbox_path.name}...")
    try:
        box = mailbox.mbox(str(mbox_path))
        for msg in box:
            try:
                subject, body = extract_body_and_subject(msg)
                full_text = f"Subject: {subject}\n\n{body}".strip() if subject else body.strip()
                if full_text:
                    records.append({
                        "text": full_text,
                        "label": label,
                        "source": source_name,
                    })
            except Exception:
                continue
    except Exception as e:
        print(f"  [ERROR] Parsing mbox failed: {e}")
    print(f"    Extracted {len(records)} messages from {source_name}.")
    return records


def load_raw_datasets() -> pd.DataFrame:
    """Download and parse all configured dataset archives."""
    all_records = []
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for label, sources in DATASET_SOURCES.items():
        print(f"\nProcessing category: {label}")
        for src in sources:
            file_path = RAW_DIR / src["filename"]
            success = download_file(src["url"], file_path)
            if not success:
                print(f"  [WARNING] Could not obtain {src['name']}. Skipping.")
                continue

            if src["type"] == "tarbz2":
                recs = parse_tarbz2(file_path, src["name"], label)
                all_records.extend(recs)
            elif src["type"] == "mbox":
                recs = parse_mbox(file_path, src["name"], label)
                all_records.extend(recs)

    df = pd.DataFrame(all_records)
    print(f"\nTotal raw emails parsed: {len(df)}")
    return df


def clean_and_balance(df: pd.DataFrame, max_per_class: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Clean dataset:
      - Drop nulls / empty text
      - Drop very short emails (< 30 chars or < 5 words)
      - Drop duplicate emails
      - Balance classes up to min_count or max_per_class
    """
    print("\n--- Cleaning Dataset ---")
    initial_count = len(df)

    # 1. Drop NA
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()

    # 2. Filter out very short emails (less than 30 characters or fewer than 5 words)
    def is_valid_email(t: str) -> bool:
        if len(t) < 30:
            return False
        words = t.split()
        return len(words) >= 5

    df = df[df["text"].apply(is_valid_email)].copy()
    print(f"  After dropping short/empty: {len(df)} (removed {initial_count - len(df)})")

    # 3. Drop duplicate texts
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    print(f"  After dropping duplicates: {len(df)} (removed {pre_dedup - len(df)})")

    # Print current distribution
    print("\nClass distribution before balancing:")
    print(df["label"].value_counts())

    # 4. Balance classes
    counts = df["label"].value_counts()
    target_count = min(counts.min(), max_per_class)
    print(f"\nBalancing each class to {target_count} samples (random_state={random_state})...")

    balanced_dfs = []
    for label in ["Legitimate", "Spam", "Phishing"]:
        sub_df = df[df["label"] == label]
        if len(sub_df) > target_count:
            sub_df = sub_df.sample(n=target_count, random_state=random_state)
        balanced_dfs.append(sub_df)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    # Shuffle
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print("\nFinal balanced distribution:")
    print(balanced_df["label"].value_counts())
    return balanced_df


def generate_eda(df: pd.DataFrame, output_dir: Path):
    """Generate exploratory data analysis charts and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = {"Legitimate": "#2ecc71", "Spam": "#f39c12", "Phishing": "#e74c3c"}

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(8, 5))
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    ax = sns.barplot(data=counts, x="Class", y="Count", palette=[palette[c] for c in counts["Class"]])
    plt.title("Email Dataset Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Email Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 6), textcoords="offset points",
                    fontweight="bold")
    plt.tight_layout()
    dist_path = output_dir / "eda_class_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"  Saved EDA: {dist_path.name}")

    # 2. Email Length Distribution (Characters and Words)
    df["char_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Boxplot / violin for word count (capped visually for readability)
    sns.boxplot(data=df, x="label", y="word_count", palette=palette, ax=axes[0], showfliers=False)
    axes[0].set_title("Word Count Distribution by Class (without extreme outliers)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Class", fontsize=11)
    axes[0].set_ylabel("Word Count", fontsize=11)

    # Character length histogram with log scale
    sns.histplot(data=df, x="char_length", hue="label", palette=palette, log_scale=True,
                 kde=True, element="step", ax=axes[1])
    axes[1].set_title("Character Length Distribution (Log Scale)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Character Length (Log Scale)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)

    plt.tight_layout()
    len_path = output_dir / "eda_length_distribution.png"
    plt.savefig(len_path, dpi=300)
    plt.close()
    print(f"  Saved EDA: {len_path.name}")

    # 3. Top Words per Class (Word Frequency Analysis)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    classes = ["Legitimate", "Spam", "Phishing"]

    # Basic stopwords list for quick EDA visualization
    basic_stop = {
        "the", "to", "and", "of", "a", "in", "for", "is", "that", "this", "you", "i", "it",
        "on", "with", "as", "have", "be", "at", "from", "by", "an", "are", "your", "all",
        "not", "we", "or", "will", "my", "has", "if", "subject", "re", "http", "https",
        "com", "net", "org", "www", "email", "mail", "message", "please", "click", "can",
        "do", "so", "no", "he", "she", "they", "me", "him", "her", "us", "our", "would",
        "there", "been", "one", "get", "more", "out", "about", "up", "was", "what", "which",
        "when", "who", "any", "some", "time", "just", "now", "know", "see", "think", "like"
    }

    eda_summary = {
        "total_samples": len(df),
        "class_counts": df["label"].value_counts().to_dict(),
        "stats_by_class": {},
        "top_keywords_by_class": {},
    }

    for idx, cls_name in enumerate(classes):
        cls_texts = df[df["label"] == cls_name]["text"]
        words = []
        for text in cls_texts:
            tokens = re.findall(r"\b[a-zA-Z]{3,15}\b", text.lower())
            words.extend([w for w in tokens if w not in basic_stop])

        word_counts = pd.Series(words).value_counts().head(15)
        eda_summary["top_keywords_by_class"][cls_name] = word_counts.head(10).to_dict()

        stats = {
            "mean_word_count": float(df[df["label"] == cls_name]["word_count"].mean()),
            "median_word_count": float(df[df["label"] == cls_name]["word_count"].median()),
            "mean_char_length": float(df[df["label"] == cls_name]["char_length"].mean()),
        }
        eda_summary["stats_by_class"][cls_name] = stats

        ax = axes[idx]
        y_pos = range(len(word_counts))
        ax.barh(y_pos, word_counts.values, color=palette[cls_name], align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(word_counts.index, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"Top 15 Words: {cls_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency", fontsize=10)

    plt.tight_layout()
    top_words_path = output_dir / "eda_top_words.png"
    plt.savefig(top_words_path, dpi=300)
    plt.close()
    print(f"  Saved EDA: {top_words_path.name}")

    # Save summary JSON
    summary_path = output_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"  Saved EDA Summary: {summary_path.name}")


def main():
    print("=" * 60)
    print("AI-BASED EMAIL PHISHING DETECTION: DATA LOADER & EDA")
    print("=" * 60)

    raw_df = load_raw_datasets()
    if raw_df.empty:
        print("[ERROR] No email data collected. Please verify network access.")
        return

    clean_df = clean_and_balance(raw_df, max_per_class=8000, random_state=42)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_df[["text", "label", "source"]].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSuccessfully saved dataset to: {OUTPUT_CSV} ({len(clean_df):,} rows)")

    print("\nGenerating Exploratory Data Analysis (EDA) reports...")
    generate_eda(clean_df, REPORTS_DIR)
    print("\nPhase 1 Data Preparation & EDA Completed Successfully!")


if __name__ == "__main__":
    main()

