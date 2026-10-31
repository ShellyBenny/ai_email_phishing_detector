"""
Feature Extraction and Rule-Based Red Flag Analysis for AI-Based Email Phishing Detection.

Detects:
1. URL Red Flags:
   - IP address instead of domain
   - URL shorteners (bit.ly, tinyurl, etc.)
   - Suspicious TLDs (.xyz, .top, .buzz, etc.)
   - '@' symbol in URL
   - Excessive subdomains (> 3 levels)
   - Very long URLs (> 75 chars)
   - Non-HTTPS URLs (plain http://)
   - Link text mismatch (HTML anchor href != display domain)
   - Lookalike domains (Levenshtein typosquatting against popular brands)
2. Urgency and Threat Language:
   - Urgency markers ("urgent", "immediately", "within 24 hours")
   - Threat / fear markers ("account suspended", "unauthorized access")
   - Credential requests ("verify your account", "confirm password")
   - Reward / bait markers ("you have won", "claim prize", "lottery")
3. Attachment Red Flags:
   - Risky executable/script extensions (.exe, .scr, .bat, .vbs, .js)
   - Archive / container formats (.zip, .iso)
   - Double file extensions (e.g. invoice.pdf.exe)
"""

from collections import defaultdict
import html
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Popular brands targeted by phishing for typosquatting / homoglyph detection
POPULAR_BRANDS = [
    "paypal", "apple", "google", "microsoft", "amazon", "netflix",
    "facebook", "instagram", "chase", "wellsfargo", "bankofamerica",
    "citibank", "dropbox", "ebay", "dhl", "fedex", "ups", "usps",
    "irs", "walmart", "linkedin", "twitter", "yahoo", "binance", "coinbase"
]

# Legitimate official domains for popular brands
BRAND_OFFICIAL_DOMAINS = {
    "paypal": ["paypal.com", "paypal-communication.com"],
    "apple": ["apple.com", "icloud.com"],
    "google": ["google.com", "gmail.com", "googlemail.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "outlook.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.de"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "chase": ["chase.com"],
    "wellsfargo": ["wellsfargo.com"],
    "bankofamerica": ["bankofamerica.com"],
    "citibank": ["citibank.com", "citi.com"],
    "dropbox": ["dropbox.com"],
    "ebay": ["ebay.com"],
    "dhl": ["dhl.com"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "usps": ["usps.com"],
    "irs": ["irs.gov"],
    "walmart": ["walmart.com"],
    "linkedin": ["linkedin.com"],
    "twitter": ["twitter.com", "x.com"],
    "yahoo": ["yahoo.com"],
    "binance": ["binance.com"],
    "coinbase": ["coinbase.com"],
}

# Known URL shortener domains
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rb.gy", "rebrand.ly", "shorturl.at",
    "goo.gl", "qr.ae", "trib.al", "tiny.cc", "soo.gd", "s.id"
}

# High-risk / suspicious top-level domains frequently abused in phishing
SUSPICIOUS_TLDS = {
    "xyz", "top", "work", "loan", "click", "country", "gq", "cf", "tk",
    "ml", "ga", "fit", "buzz", "rest", "support", "stream", "gdn", "date",
    "faith", "racing", "party", "accountant", "review", "download", "men"
}

# Risky attachment extensions
RISKY_EXTENSIONS = {
    "exe", "scr", "bat", "cmd", "vbs", "vbe", "js", "jse", "wsf", "wsh",
    "ps1", "msi", "msp", "hta", "cpl", "jar", "reg", "pif", "gadget",
    "docm", "xlsm", "pptm", "iso", "img"
}

ARCHIVE_EXTENSIONS = {"zip", "rar", "7z", "tar", "gz", "bz2", "xz"}

# Urgency and threat lexicon categories
URGENCY_PATTERNS = {
    "Urgency": [
        r"\burgent(?:ly)?\b",
        r"\bimmediat(?:e|ely)\b",
        r"\bwithin\s+\d+\s+(?:hours?|mins?|minutes?|days?)\b",
        r"\bimmediate\s+action\s+required\b",
        r"\blimited\s+time\b",
        r"\bact\s+now\b",
        r"\bexpires?\s+today\b",
        r"\bdeadline\b",
        r"\bhvrry\b",
        r"\bdo\s+not\s+delay\b",
        r"\basap\b",
        r"\binstant(?:ly)?\b",
    ],
    "Threat / Fear": [
        r"\baccount\s+(?:suspended|suspended\b|terminated|deactivated|closed|frozen)\b",
        r"\bsuspended\b",
        r"\bunauthorized\s+(?:access|login|activity|transaction)\b",
        r"\bsecurity\s+(?:alert|breach|warning)\b",
        r"\block(?:ed)?\s+out\b",
        r"\blegal\s+(?:action|notice|proceedings|charges)\b",
        r"\blawsuit\b",
        r"\bfailure\s+to\s+respond\b",
        r"\bwill\s+be\s+deleted\b",
        r"\bcompromised\b",
        r"\bfraud\s+alert\b",
    ],
    "Credential Request": [
        r"\bverify\s+your\s+(?:account|identity|email|details|credentials|information)\b",
        r"\bconfirm\s+your\s+(?:password|pin|identity|details|credentials)\b",
        r"\benter\s+(?:your\s+)?(?:password|pin|credentials)\b",
        r"\bupdate\s+your\s+(?:billing|payment|account|card)\s+info(?:rmation)?\b",
        r"\bvalidate\s+your\s+(?:identity|account|credentials)\b",
        r"\bclick\s+here\s+to\s+(?:login|log\s+in|verify|update|unlock|restore)\b",
        r"\bclick\s+here\b",
        r"\bsocial\s+security\s+number\b",
        r"\bssn\b",
        r"\blogin\s+to\s+(?:restore|recover|unlock)\b",
    ],
    "Reward / Bait": [
        r"\byou\s+have\s+won\b",
        r"\bclaim\s+your\s+(?:prize|reward|gift|refund|funds)\b",
        r"\blottery\b",
        r"\binheritance\b",
        r"\b(?:million|thousand)\s+dollars\b",
        r"\bexclusive\s+(?:offer|reward|gift)\b",
        r"\bfree\s+(?:gift|money|bitcoin|crypto|card)\b",
        r"\bbonus\s+credit\b",
        r"\bwire\s+transfer\s+of\b",
    ],
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def extract_links_and_text(raw_email: str) -> list[dict]:
    """
    Extract links with their display anchor text if HTML,
    plus plain-text URLs detected via regex.
    """
    links = []
    # 1. HTML anchor tags
    if "<a " in raw_email.lower():
        try:
            soup = BeautifulSoup(raw_email, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "").strip()
                text = a.get_text(separator=" ").strip()
                if href and not href.startswith("mailto:") and not href.startswith("javascript:"):
                    links.append({"href": href, "anchor_text": text, "source": "html_anchor"})
        except Exception:
            pass

    # 2. Plain text URLs
    url_pattern = re.compile(r"(https?://[^\s<>\"')]+|www\.[^\s<>\"')]+)", re.IGNORECASE)
    for match in url_pattern.finditer(raw_email):
        url = match.group(1).rstrip(".,;!?:")
        # Check if already added
        if not any(l["href"] == url for l in links):
            links.append({"href": url, "anchor_text": "", "source": "plain_text"})

    return links


def check_brand_lookalike(domain: str) -> dict | None:
    """
    Check if domain is a lookalike/typosquatted version of a well-known brand.
    e.g., paypa1.com, micros0ft.net, secure-apple.org
    """
    clean_domain = domain.lower()
    # Strip common prefixes/suffixes
    parts = clean_domain.split(".")
    if len(parts) >= 2:
        main_part = parts[-2]
    else:
        main_part = clean_domain

    # Homoglyph replacements
    homoglyphs = {"0": "o", "1": "l", "vv": "w", "rn": "m"}
    normalized_main = main_part
    for k, v in homoglyphs.items():
        normalized_main = normalized_main.replace(k, v)

    for brand in POPULAR_BRANDS:
        official_domains = BRAND_OFFICIAL_DOMAINS.get(brand, [f"{brand}.com"])
        if any(clean_domain.endswith(off) for off in official_domains):
            continue  # Legitimate domain for brand

        # Direct homoglyph hit
        if brand in normalized_main and brand not in main_part:
            return {
                "brand": brand,
                "detected_in": domain,
                "reason": f"Homoglyph character substitution disguising '{brand}'",
            }

        # Substring impersonation: e.g. paypal-security-update.com
        if brand in main_part and not any(clean_domain == off for off in official_domains):
            return {
                "brand": brand,
                "detected_in": domain,
                "reason": f"Suspicious brand keyword '{brand}' found in third-party domain",
            }

        # Small edit distance (1 or 2 typos)
        dist = levenshtein_distance(main_part, brand)
        if 1 <= dist <= 2 and len(main_part) >= len(brand) - 1:
            return {
                "brand": brand,
                "detected_in": domain,
                "reason": f"Lookalike domain closely misspelling '{brand}' (edit distance {dist})",
            }

    return None


def analyze_urls(raw_email: str) -> dict:
    """
    Inspect all URLs and HTML links for phishing red flags.
    """
    links = extract_links_and_text(raw_email)
    red_flags = []
    analyzed_links = []
    ip_pattern = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

    for item in links:
        url = item["href"]
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
        path = parsed.path
        item_flags = []

        # 1. IP address instead of domain
        if ip_pattern.match(hostname):
            item_flags.append(f"Uses raw IP address ({hostname}) instead of domain")

        # 2. URL Shortener
        if hostname in URL_SHORTENERS:
            item_flags.append(f"Uses URL shortener service ({hostname}) to disguise destination")

        # 3. Suspicious TLD
        tld = hostname.split(".")[-1] if "." in hostname else ""
        if tld in SUSPICIOUS_TLDS:
            item_flags.append(f"Registered under high-risk TLD (.{tld}) commonly used for fraud")

        # 4. '@' in URL
        if "@" in url:
            item_flags.append("Contains '@' symbol in URL (often masks true destination host)")

        # 5. Excessive subdomains
        domain_parts = [p for p in hostname.split(".") if p]
        if len(domain_parts) > 4:
            item_flags.append(f"Excessive subdomains ({len(domain_parts) - 2} levels): {hostname}")

        # 6. Very long URL
        if len(url) > 75:
            item_flags.append(f"Abnormally long URL ({len(url)} characters)")

        # 7. Non-HTTPS link
        if url.lower().startswith("http://"):
            item_flags.append("Insecure protocol (HTTP instead of encrypted HTTPS)")

        # 8. Link text mismatch in HTML emails
        anchor = item.get("anchor_text", "").strip()
        if anchor and ("http://" in anchor or "https://" in anchor or "www." in anchor):
            anchor_parsed = urlparse(anchor if "://" in anchor else "http://" + anchor)
            anchor_host = (anchor_parsed.hostname or "").lower()
            if anchor_host and anchor_host != hostname and not hostname.endswith("." + anchor_host):
                item_flags.append(
                    f"Link text deceptive mismatch: Displayed as '{anchor_host}' but points to '{hostname}'"
                )

        # 9. Lookalike domain
        if hostname:
            lookalike = check_brand_lookalike(hostname)
            if lookalike:
                item_flags.append(f"Lookalike brand spoofing: {lookalike['reason']}")

        if item_flags:
            red_flags.extend(item_flags)

        analyzed_links.append({
            "url": url,
            "display_text": anchor,
            "hostname": hostname,
            "flags": item_flags,
        })

    # Deduplicate red flags
    unique_flags = list(dict.fromkeys(red_flags))

    return {
        "total_urls_found": len(links),
        "suspicious_url_count": sum(1 for l in analyzed_links if l["flags"]),
        "red_flags": unique_flags,
        "links_detail": analyzed_links,
    }


def analyze_urgency_and_threats(text: str) -> dict:
    """
    Detect urgency, threat, credential theft, and reward bait keywords/phrases.
    """
    findings = defaultdict(list)
    total_matches = 0
    clean_sample = text.lower()

    for category, patterns in URGENCY_PATTERNS.items():
        for pat in patterns:
            for match in re.finditer(pat, clean_sample, re.IGNORECASE):
                phrase = match.group(0).strip()
                if phrase not in findings[category]:
                    findings[category].append(phrase)
                total_matches += 1

    return {
        "total_triggers": total_matches,
        "categories_triggered": len(findings),
        "breakdown": dict(findings),
    }


def analyze_attachments(text_or_filenames: list[str] | str) -> dict:
    """
    Detect risky, executable, or double extensions in attachment filenames
    or mentioned file downloads.
    """
    filenames = []
    if isinstance(text_or_filenames, str):
        # Extract potential filenames mentioned in text (e.g. invoice.pdf.exe)
        found = re.findall(r"\b[A-Za-z0-9_\-.]+\.[A-Za-z0-9]{2,5}\b", text_or_filenames)
        filenames.extend(found)
    else:
        filenames.extend(text_or_filenames)

    flags = []
    suspicious_files = []

    for fn in filenames:
        fn_clean = fn.strip().lower()
        parts = fn_clean.split(".")
        if len(parts) < 2:
            continue

        ext = parts[-1]
        is_suspicious = False

        # Double extension check: e.g. document.pdf.exe
        if len(parts) >= 3:
            second_last = parts[-2]
            if second_last in {"pdf", "doc", "docx", "xls", "xlsx", "txt", "jpg", "png"}:
                if ext in RISKY_EXTENSIONS or ext in ARCHIVE_EXTENSIONS:
                    flags.append(f"Dangerous double extension detected: '{fn}' (.pdf/.doc disguised executable)")
                    is_suspicious = True

        # Risky executable/script extension
        if ext in RISKY_EXTENSIONS:
            flags.append(f"Executable or dangerous script attachment: '{fn}' (.{ext})")
            is_suspicious = True

        # Archive file
        if ext in ARCHIVE_EXTENSIONS and not is_suspicious:
            flags.append(f"Compressed archive attachment: '{fn}' (often conceals payload)")
            is_suspicious = True

        if is_suspicious:
            suspicious_files.append(fn)

    return {
        "suspicious_attachments_count": len(suspicious_files),
        "suspicious_files": suspicious_files,
        "red_flags": list(dict.fromkeys(flags)),
    }


def extract_all_features(email_text: str, attachment_list: list[str] | None = None) -> dict:
    """
    Run full suite of rule-based feature extractors on the email.
    """
    url_results = analyze_urls(email_text)
    urgency_results = analyze_urgency_and_threats(email_text)
    att_input = attachment_list if attachment_list else email_text
    att_results = analyze_attachments(att_input)

    return {
        "url_analysis": url_results,
        "urgency_analysis": urgency_results,
        "attachment_analysis": att_results,
    }


if __name__ == "__main__":
    test_sample = """
    From: "PayPal Support" <support@paypa1-security.com>
    Subject: URGENT: Your PayPal Account Has Been Suspended!
    
    Dear customer,
    We detected unauthorized access to your account. You must verify your account within 24 hours
    or your account will be deleted permanently.
    
    Click here to confirm password: <a href="http://192.168.1.100/paypal-login">https://paypal.com/verify</a>
    
    Please also review your attached statement: invoice_9921.pdf.exe
    """
    print("Testing Feature Extraction:")
    res = extract_all_features(test_sample)
    print("\nURL Red Flags:", res["url_analysis"]["red_flags"])
    print("\nUrgency Triggers:", res["urgency_analysis"]["breakdown"])
    print("\nAttachment Red Flags:", res["attachment_analysis"]["red_flags"])

