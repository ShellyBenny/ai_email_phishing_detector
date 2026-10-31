"""
Email Header Security Analyzer for AI-Based Email Phishing Detection.
Analyzes RFC 5322 email headers from uploaded .eml files or pasted raw email data.

Checks:
1. From vs Reply-To domain mismatch
2. From vs Return-Path envelope mismatch
3. Display Name Spoofing (brand impersonation or fake email in display name)
4. SPF, DKIM, DMARC authentication verdicts
5. Suspicious Received chain & relay hops

Gracefully handles cases where headers are absent (e.g. pasted email body only).
"""

import email
from email import policy
from email.header import decode_header
from email.utils import parseaddr
import re
from urllib.parse import urlparse

# Popular brands monitored for display-name spoofing
TARGETED_BRANDS = {
    "paypal": ["paypal.com", "paypal-communication.com"],
    "apple": ["apple.com", "icloud.com"],
    "google": ["google.com", "gmail.com"],
    "microsoft": ["microsoft.com", "office.com", "outlook.com", "live.com"],
    "amazon": ["amazon.com", "amazon.co.uk"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com"],
    "wellsfargo": ["wellsfargo.com"],
    "bankofamerica": ["bankofamerica.com"],
    "citibank": ["citibank.com", "citi.com"],
    "irs": ["irs.gov"],
    "dhl": ["dhl.com"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "usps": ["usps.com"],
}


def decode_header_value(header_val: str) -> str:
    """Safely decode RFC 2047 encoded email header strings."""
    if not header_val:
        return ""
    try:
        decoded_fragments = decode_header(header_val)
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
        return str(header_val).strip()


def extract_domain(email_addr: str) -> str:
    """Extract domain portion of an email address."""
    if not email_addr or "@" not in email_addr:
        return ""
    return email_addr.split("@")[-1].strip().lower()


def parse_email_message(content: str | bytes):
    """Parse raw string or bytes into an EmailMessage object."""
    if isinstance(content, str):
        # Quick check if it looks like an email message with headers
        lines = content.splitlines()[:20]
        has_rfc_headers = any(re.match(r"^(?:from|to|subject|date|received|message-id):", l, re.IGNORECASE) for l in lines)
        if not has_rfc_headers:
            return None, False
        msg = email.message_from_string(content, policy=policy.compat32)
        return msg, True
    elif isinstance(content, bytes):
        msg = email.message_from_bytes(content, policy=policy.compat32)
        return msg, True
    return None, False


def parse_authentication_results(msg) -> dict:
    """
    Parse SPF, DKIM, and DMARC results from Authentication-Results,
    Received-SPF, or X-DKIM headers.
    """
    results = {
        "spf": "not_present",
        "dkim": "not_present",
        "dmarc": "not_present",
        "raw_headers": [],
    }

    auth_headers = msg.get_all("Authentication-Results", [])
    rec_spf = msg.get_all("Received-SPF", [])

    for h in auth_headers:
        val = str(h).lower()
        results["raw_headers"].append(val)
        if "spf=pass" in val:
            results["spf"] = "pass"
        elif "spf=fail" in val or "spf=softfail" in val:
            results["spf"] = "fail"

        if "dkim=pass" in val:
            results["dkim"] = "pass"
        elif "dkim=fail" in val:
            results["dkim"] = "fail"

        if "dmarc=pass" in val:
            results["dmarc"] = "pass"
        elif "dmarc=fail" in val:
            results["dmarc"] = "fail"

    for h in rec_spf:
        val = str(h).lower()
        results["raw_headers"].append(val)
        if val.startswith("pass") or "result=pass" in val:
            results["spf"] = "pass"
        elif val.startswith("fail") or val.startswith("softfail"):
            results["spf"] = "fail"

    return results


def parse_received_chain(msg) -> list[dict]:
    """Extract and parse chronological hops from Received headers."""
    received_headers = msg.get_all("Received", [])
    hops = []
    ip_pattern = re.compile(r"\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]")

    for idx, r_val in enumerate(received_headers):
        val_str = " ".join(str(r_val).split())
        ips = ip_pattern.findall(val_str)
        from_part = ""
        by_part = ""

        from_match = re.search(r"from\s+([^\s;]+)", val_str, re.IGNORECASE)
        if from_match:
            from_part = from_match.group(1)

        by_match = re.search(r"by\s+([^\s;]+)", val_str, re.IGNORECASE)
        if by_match:
            by_part = by_match.group(1)

        hops.append({
            "hop": idx + 1,
            "from": from_part,
            "by": by_part,
            "ips": ips,
            "raw": val_str[:120] + ("..." if len(val_str) > 120 else ""),
        })

    return hops


def analyze_headers(raw_content: str | bytes) -> dict:
    """
    Perform complete header security analysis.
    Gracefully returns headers_present=False if text is a pasted body.
    """
    msg, has_headers = parse_email_message(raw_content)

    if not has_headers or msg is None:
        return {
            "headers_present": False,
            "status": "Skipped (Pasted email body without RFC 5322 headers)",
            "sender_info": {},
            "auth_results": {},
            "received_chain": [],
            "red_flags": [],
            "header_risk_penalty": 0.0,
        }

    # Extract primary identity headers
    from_raw = decode_header_value(msg.get("From", ""))
    reply_to_raw = decode_header_value(msg.get("Reply-To", ""))
    return_path_raw = decode_header_value(msg.get("Return-Path", ""))
    subject = decode_header_value(msg.get("Subject", ""))
    date = decode_header_value(msg.get("Date", ""))
    message_id = decode_header_value(msg.get("Message-ID", ""))

    display_name, from_email = parseaddr(from_raw)
    _, reply_to_email = parseaddr(reply_to_raw)
    _, return_path_email = parseaddr(return_path_raw)

    from_domain = extract_domain(from_email)
    reply_to_domain = extract_domain(reply_to_email)
    return_path_domain = extract_domain(return_path_email)

    red_flags = []
    penalty = 0.0

    # 1. From vs Reply-To Mismatch
    if reply_to_domain and from_domain and reply_to_domain != from_domain:
        # Check if subdomains of same root
        if not (reply_to_domain.endswith("." + from_domain) or from_domain.endswith("." + reply_to_domain)):
            red_flags.append(
                f"From vs Reply-To mismatch: Sender is '{from_email}', but replies route to '{reply_to_email}'"
            )
            penalty += 25.0

    # 2. From vs Return-Path Mismatch
    if return_path_domain and from_domain and return_path_domain != from_domain:
        if not (return_path_domain.endswith("." + from_domain) or from_domain.endswith("." + return_path_domain)):
            red_flags.append(
                f"From vs Return-Path mismatch: Envelope sender '{return_path_email}' does not match header '{from_email}'"
            )
            penalty += 20.0

    # 3. Display Name Spoofing
    disp_lower = display_name.lower()
    # Check if display name contains an embedded email address
    embedded_email_match = re.search(r"[\w.-]+@[\w.-]+\.\w+", display_name)
    if embedded_email_match:
        embedded_email = embedded_email_match.group(0).lower()
        if embedded_email != from_email.lower():
            red_flags.append(
                f"Display name deception: Display name contains email '{embedded_email}', but actual sender is '{from_email}'"
            )
            penalty += 30.0

    # Check if display name impersonates a major brand while sending from generic domain
    for brand, legit_domains in TARGETED_BRANDS.items():
        if brand in disp_lower:
            if not any(from_domain.endswith(d) for d in legit_domains):
                red_flags.append(
                    f"Brand impersonation in Display Name: Claims to be '{display_name}' ({brand.capitalize()}), but sent from untrusted domain '{from_domain}'"
                )
                penalty += 35.0
                break

    # 4. Authentication (SPF / DKIM / DMARC)
    auth_results = parse_authentication_results(msg)
    if auth_results["spf"] == "fail":
        red_flags.append("SPF verification failed (sender IP unauthorized for sending domain)")
        penalty += 25.0
    if auth_results["dkim"] == "fail":
        red_flags.append("DKIM cryptographic signature verification failed (message altered or forged)")
        penalty += 25.0
    if auth_results["dmarc"] == "fail":
        red_flags.append("DMARC policy enforcement failed (domain alignment check failed)")
        penalty += 30.0

    # 5. Received Chain
    hops = parse_received_chain(msg)
    # Check if the message contains full transport/envelope headers (e.g. Return-Path, DKIM, Auth-Results, or many headers)
    has_full_transport_headers = bool(
        msg.get_all("Authentication-Results")
        or msg.get_all("Received-SPF")
        or msg.get_all("DKIM-Signature")
        or (msg.get("Return-Path") and msg.get("Message-ID"))
        or (len(msg.keys()) >= 8 and msg.get("Message-ID"))
    )

    if len(hops) == 0 and has_full_transport_headers:
        red_flags.append("Suspicious absence of mail transfer Received hops in header")
        penalty += 15.0

    # Normalize penalty to 0 - 100
    penalty = min(penalty, 100.0)

    return {
        "headers_present": True,
        "status": "Headers parsed successfully",
        "sender_info": {
            "from_raw": from_raw,
            "display_name": display_name,
            "from_email": from_email,
            "from_domain": from_domain,
            "reply_to_email": reply_to_email,
            "return_path_email": return_path_email,
            "subject": subject,
            "date": date,
            "message_id": message_id,
        },
        "auth_results": auth_results,
        "received_chain": hops,
        "red_flags": red_flags,
        "header_risk_penalty": penalty,
    }


if __name__ == "__main__":
    eml_sample = """From: "PayPal Account Security" <support@fake-alert-desk.net>
Reply-To: attacker-inbox@mail-drop.ru
Return-Path: bounce@marketing-blast.xyz
Subject: Urgent Security Alert: Unauthorized Login Attempt
Date: Fri, 30 Oct 2026 21:00:00 +0000
Message-ID: <1234567@fake-alert-desk.net>
Authentication-Results: mx.google.com; spf=fail (google.com: domain does not designate 185.220.101.5 as permitted sender); dkim=fail; dmarc=fail
Received: from mail.fake-alert-desk.net ([185.220.101.5]) by mx.google.com with ESMTP id abc123; Fri, 30 Oct 2026 21:00:01 +0000

Dear customer, please verify your credentials.
"""
    print("Testing Header Analysis on Spoofed Email:")
    res = analyze_headers(eml_sample)
    print("Headers Present:", res["headers_present"])
    print("Red Flags:", res["red_flags"])
    print("Penalty Score:", res["header_risk_penalty"])

    print("\nTesting Pasted Body Without Headers:")
    pasted_sample = "Hey team, remember our standup meeting is moved to 3 PM today. Thanks!"
    pasted_res = analyze_headers(pasted_sample)
    print("Headers Present:", pasted_res["headers_present"])
    print("Status:", pasted_res["status"])

