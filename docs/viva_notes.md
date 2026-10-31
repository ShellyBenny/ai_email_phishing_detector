# Viva Voce Examination Notes & Technical Defense Guide
## Project: AI-Based Email Phishing Detection System

This document is structured specifically to prepare you for comprehensive viva voce examinations, technical interviews, and project defense presentations.

---

## 1. High-Level Architecture & Problem Formulation

### Q1: What is the core problem and why is binary classification (Ham vs. Spam) insufficient?
> **Answer**:
> Traditional spam filters treat email classification as binary: *Legitimate (Ham)* vs. *Spam*. This conflates two fundamentally different categories:
> 1. **Commercial Spam**: Unsolicited advertising, marketing newsletters, and promotions. While annoying and bandwidth-consuming, spam does not directly attempt to compromise infrastructure, steal credentials, or commit identity theft.
> 2. **Phishing / Social Engineering**: Targeted cyberattacks designed to deceive victims into surrendering banking credentials, corporate SSO passwords, or executing malware.
>
> In our system, we formulate the problem as **3-class classification (Legitimate, Spam, Phishing)**. This enables tailored threat responses:
> - Legitimate $\to$ Normal inbox delivery with standard vigilance.
> - Spam $\to$ Auto-archive, junk folder routing, or unsubscribe advice.
> - Phishing $\to$ High-priority alert banner, URL blocking, incident response escalation, and immediate password reset guidance.

---

## 2. NLP Preprocessing & Feature Engineering

### Q2: Why did you choose TF-IDF over simple Bag-of-Words (CountVectorizer) or Deep Learning (Word2Vec/BERT)?
> **Answer**:
> 1. **Explainability**: In high-stakes cybersecurity and academic viva evaluations, "black box" deep learning models (e.g., Transformers) cannot easily explain why a specific email was flagged without complex, slow post-hoc tools like SHAP or LIME. With TF-IDF and linear models, each feature has an exact mathematical contribution weight ($x_j \cdot w_{c, j}$) directly mapped back to intelligible vocabulary terms.
> 2. **Computational Efficiency & Latency**: TF-IDF transforms and scores an email in under 5 milliseconds on a standard CPU, requires zero GPU infrastructure, and deploys as a lightweight container (< 300MB).
> 3. **Sublinear Term Frequency**: We configured `sublinear_tf=True`, which replaces raw term frequency $\text{tf}$ with $1 + \log(\text{tf})$. This prevents an email that spams the word "viagra" or "account" 50 times from having 50x the weight of an email that mentions it twice.
> 4. **N-gram Range**: Using unigrams and bigrams (`ngram_range=(1, 2)`) allows the model to capture critical multi-word contextual phrases such as `"account suspended"`, `"verify identity"`, `"click here"`, and `"unauthorized access"` that individual words alone cannot convey.

### Q3: What is the purpose of entity marker tokens (`urltoken`, `emailtoken`, `currencytoken`, `numbertoken`)?
> **Answer**:
> If raw URLs like `http://192.168.1.1/login?id=99281` are passed into a standard tokenizer, they fragment into arbitrary strings or single-use numbers that inflate the vocabulary dictionary without generalizing. By replacing URLs with `urltoken`, emails with `emailtoken`, currency amounts with `currencytoken`, and numbers with `numbertoken`:
> - The model learns the **structural behavioral density** of emails (e.g., phishing emails typically exhibit a high density of URLs and currency/number tokens).
> - The vocabulary size is kept compact (8,000 features), preventing out-of-vocabulary degradation.

---

## 3. Classifiers, Calibration & Evaluation Metrics

### Q4: Why did Linear SVM outperform other models, and why was `CalibratedClassifierCV` necessary?
> **Answer**:
> - **Performance**: In high-dimensional sparse text classification (8,000 TF-IDF features), text documents are almost always linearly separable. Support Vector Machines find the maximum-margin hyperplane separating classes, providing superior generalization compared to generative models like Naive Bayes.
> - **The Calibration Necessity**: Standard Linear SVM (`LinearSVC`) is a non-probabilistic margin classifier. It outputs an unconstrained geometric distance $f(x) \in (-\infty, +\infty)$ from the decision boundary. Because our application requires confidence percentages and uses the probability of phishing $P(\text{Phishing})$ in the Risk Score formula, we applied **Platt Scaling** via `CalibratedClassifierCV(method='sigmoid')`. This fits a cross-validated logistic sigmoid over the SVM decision values, producing mathematically valid posterior probabilities in $[0, 1]$.

### Q5: Why is Macro-F1 used for model selection rather than Accuracy or Micro-F1?
> **Answer**:
> In multi-class classification:
> - **Accuracy** treats all correctly predicted samples equally, which hides poor performance on critical minority classes in imbalanced scenarios.
> - **Micro-F1** aggregates global true positives, false positives, and false negatives, which numerically equals overall accuracy in single-label multi-class tasks.
> - **Macro-F1** calculates the unweighted arithmetic mean of F1-scores across all three classes:
>   $$\text{Macro-F1} = \frac{\text{F1}_{\text{Legitimate}} + \text{F1}_{\text{Spam}} + \text{F1}_{\text{Phishing}}}{3}$$
> This forces the model selection process to value detecting Phishing, Spam, and Legitimate emails equally. A model cannot achieve a high Macro-F1 by simply favoring one dominant class.

### Q6: Explain the trade-off between Precision and Recall in phishing detection.
> **Answer**:
> - **Precision (Phishing)**: $\frac{TP}{TP + FP}$ — Out of all emails the system flagged as Phishing, how many were truly phishing?
>   - *Low precision* $\to$ Legitimate emails (e.g., invoices, HR emails) are falsely quarantined as phishing (high business friction).
> - **Recall (Phishing)**: $\frac{TP}{TP + FN}$ — Out of all actual phishing attacks, how many did the system catch?
>   - *Low recall* $\to$ Phishing attacks slip into user inboxes (high security catastrophe risk).
>
> In our pipeline, we tuned hyperparameters via 5-fold cross-validation to maximize Phishing Recall while maintaining high Precision, and we introduced the **Sensitivity Slider** in the UI so security administrators can tune the threshold according to organizational risk tolerance.

---

## 4. Heuristics, Header Security & Risk Scoring

### Q7: How does your system detect lookalike domains (e.g., `paypa1.com`, `amaz0n.xyz`)?
> **Answer**:
> In `src/features.py`, we implement a two-stage typosquatting detector:
> 1. **Homoglyph Normalization**: We replace deceptive character substitutions (`0` $\to$ `o`, `1` $\to$ `l`, `rn` $\to$ `m`, `vv` $\to$ `w`). If the normalized string matches a monitored brand but the original does not, it triggers a homoglyph alert.
> 2. **Levenshtein Edit Distance**: We compute the minimum number of single-character edits (insertions, deletions, substitutions) between the extracted domain and a dictionary of high-value brands (PayPal, Apple, Google, Amazon, Chase, etc.). If the edit distance is 1 or 2 and the domain is not in the brand's verified domain list, it is flagged as lookalike typosquatting.

### Q8: How does the system handle emails without RFC 5322 headers?
> **Answer**:
> In real-world usage, users frequently paste only the body of an email into a web portal. In `src/headers.py`, we parse RFC 5322 headers (`From:`, `Subject:`, `Date:`, `Received:`). If fewer than two standard headers exist, the analyzer returns `headers_present=False`.
> The Risk Scoring formula dynamically switches weight profiles:
> - **With Headers**: Model (40%), URLs (25%), Urgency (15%), Attachments (10%), Headers (10%).
> - **Without Headers**: Model (45%), URLs (25%), Urgency (20%), Attachments (10%), Headers (0%).
> This ensures users are not penalized for headers that were simply omitted during copy-pasting.

---

## 5. Dataset Provenance, Corpus Bias & Honest Evaluation

### Q9: What datasets were used, and what are their limitations?
> **Answer**:
> - **Corpora**: Apache SpamAssassin (2003–2005) for Legitimate and Spam, and Jose Nazario's Phishing Corpus for Phishing.
> - **Corpus Bias & Limitations**:
>   1. **Temporal Gap**: These benchmark corpora originate from the mid-2000s. While foundational security concepts remain identical (urgency, account suspension, fake links), modern phishing incorporates newer vectors such as QR code phishing ("quishing"), multi-stage MFA session hijacking, and LLM-generated spear phishing.
>   2. **Format Differences**: Older ham datasets contain plain-text mailing list discussions (e.g., Linux kernel mailing list), whereas modern business communications use rich HTML formatting and cloud collaboration links.
>   3. **Language Scope**: Stopwords and lemmatization are optimized for English; non-English emails require language detection and specialized multilingual tokenizers.

### Q10: How did the model perform on the 15 fresh, unseen test emails (Phase 7)?
> **Answer**:
> The model achieved **66.7% overall accuracy (10/15)** on realistic, modern fictional emails without any custom retraining:
> - **Spam**: 5/5 (100%) — Commercial marketing, furniture clearance, replica watches, and webinar sales were perfectly classified.
> - **Phishing**: 3/5 (60%) — Chase Bank alerts, Microsoft 365 password expiry, and IRS refund scams were identified. The USPS package delivery notice was caught by the **URL heuristics** (Risk Score 38.4 [Medium Risk]) despite the text model being split between Spam and Legitimate. The fake job offer was classified as Spam due to generic employment keywords.
> - **Legitimate**: 2/5 (40%) — Academic feedback and family dinner invitations passed easily, while brief corporate IT maintenance notices were slightly over-flagged due to technical words like `"cloud"` and `"attached"`.
>
> This demonstrates the essential role of our **defense-in-depth architecture**: even when the text ML classifier is uncertain, the heuristic rules (URL scrutiny, header checks, urgency detectors) elevate the overall Risk Score to protect the user.

