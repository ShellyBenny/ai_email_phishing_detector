"""
Model Training, Cross-Validation, Hyperparameter Tuning, and Evaluation Pipeline
for AI-Based Email Phishing Detection.

Models Evaluated:
1. Multinomial Naive Bayes (Core Classifier 1)
2. Logistic Regression (Core Classifier 2)
3. Linear SVM with Probability Calibration (Comparison Model 1)
4. Random Forest (Comparison Model 2)

Artifacts saved:
- models/vectorizer.joblib
- models/best_model.joblib
- models/naive_bayes.joblib
- models/logistic_regression.joblib
- models/linear_svm.joblib
- models/random_forest.joblib
- models/metrics.json
- reports/confusion_matrix_{model}.png
- reports/confusion_matrices_all.png
- reports/roc_curves_{model}.png
- reports/roc_comparison.png
- reports/model_comparison.png
- reports/errors.csv
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_config"
from datetime import datetime
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

# Adjust path for imports
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocess import clean_text

DATA_PATH = BASE_DIR / "data" / "emails.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CLASS_NAMES = ["Legitimate", "Phishing", "Spam"]
RANDOM_STATE = 42


def load_and_preprocess_data():
    """Load emails.csv and apply preprocessing."""
    print("\n[Step 1/6] Loading and Preprocessing Data...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run src/data_loader.py first.")

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} raw emails from {DATA_PATH.name}")

    # Check for cached cleaned column or clean fresh
    cleaned_cache = BASE_DIR / "data" / "emails_cleaned.csv"
    if cleaned_cache.exists():
        print(f"  Loading preprocessed cache from {cleaned_cache.name}...")
        df = pd.read_csv(cleaned_cache)
    else:
        print("  Applying clean_text() to dataset (cleaning HTML, URLs, entities, lemmatizing)...")
        df["clean_text"] = df["text"].astype(str).apply(clean_text)
        # Drop rows that became empty after cleaning
        df = df[df["clean_text"].str.strip().str.len() > 0].copy()
        df.to_csv(cleaned_cache, index=False)
        print(f"  Cached cleaned dataset to {cleaned_cache.name} ({len(df):,} valid samples)")

    return df


def split_and_vectorize(df):
    """Perform 80/20 Stratified train-test split and TF-IDF vectorization."""
    print("\n[Step 2/6] Splitting (80/20 Stratified) and Vectorizing (TF-IDF)...")
    X = np.array(df["clean_text"].astype(str).tolist(), dtype=object)
    y = np.array(df["label"].astype(str).tolist(), dtype=object)
    indices = np.arange(len(df), dtype=np.int64)

    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train set: {len(X_train_raw):,} samples | Test set: {len(X_test_raw):,} samples")

    # TF-IDF with unigrams + bigrams and sublinear_tf=True
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=8000,
        min_df=3,
    )
    print("  Fitting TF-IDF vectorizer (unigrams + bigrams, sublinear_tf=True, max_features=8000)...")
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"  TF-IDF Feature vocabulary size: {len(vectorizer.vocabulary_):,}")

    return vectorizer, X_train, X_test, y_train, y_test, idx_test, df


def build_and_tune_models(X_train, y_train):
    """
    Train and tune 4 models with GridSearchCV using 5-fold Stratified CV.
    Models:
      1. Multinomial Naive Bayes (Core 1)
      2. Logistic Regression (Core 2)
      3. Linear SVM (Comparison 1, calibrated for probabilities)
      4. Random Forest (Comparison 2)
    """
    print("\n[Step 3/6] Hyperparameter Tuning with GridSearchCV (5-Fold Stratified CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_configs = {
        "Naive Bayes": {
            "estimator": MultinomialNB(),
            "params": {"alpha": [0.1, 0.5, 1.0]},
            "is_calibrated": False,
        },
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"),
            "params": {"C": [0.5, 1.0, 5.0]},
            "is_calibrated": False,
        },
        "Linear SVM": {
            "estimator": LinearSVC(random_state=RANDOM_STATE, max_iter=2000),
            "params": {"C": [0.1, 1.0, 3.0]},
            "is_calibrated": True,  # Will be calibrated with CalibratedClassifierCV
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {"n_estimators": [100, 150], "max_depth": [25, None]},
            "is_calibrated": False,
        },
    }

    trained_models = {}

    for name, config in model_configs.items():
        print(f"  Tuning {name}...")
        grid = GridSearchCV(
            config["estimator"],
            config["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"    Best params: {grid.best_params_} (5-fold CV Macro-F1: {grid.best_score_:.4f})")

        # Calibrate Linear SVM to ensure reliable probabilities for confidence scores
        if config["is_calibrated"]:
            print("    Applying CalibratedClassifierCV (isotonic) to Linear SVM for probability scores...")
            calibrated = CalibratedClassifierCV(estimator=best_model, cv=3, method="sigmoid")
            calibrated.fit(X_train, y_train)
            final_model = calibrated
        else:
            final_model = best_model

        trained_models[name] = {
            "model": final_model,
            "best_params": grid.best_params_,
            "cv_macro_f1": float(grid.best_score_),
            "cv_results": grid.cv_results_,
        }

    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate all models on:
      - Accuracy
      - Per-class Precision, Recall, F1
      - Macro-F1
      - Confusion Matrices
      - One-vs-Rest ROC curves and AUC
    """
    print("\n[Step 4/6] Evaluating Models on Held-out Test Set...")
    eval_results = {}
    classes = np.array(CLASS_NAMES)
    y_test_bin = label_binarize(y_test, classes=classes)

    for name, data in trained_models.items():
        model = data["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)

        # Per-class metrics
        precisions = precision_score(y_test, y_pred, labels=CLASS_NAMES, average=None)
        recalls = recall_score(y_test, y_pred, labels=CLASS_NAMES, average=None)
        f1s = f1_score(y_test, y_pred, labels=CLASS_NAMES, average=None)

        per_class = {}
        for idx, cls in enumerate(CLASS_NAMES):
            per_class[cls] = {
                "precision": float(precisions[idx]),
                "recall": float(recalls[idx]),
                "f1_score": float(f1s[idx]),
            }

        # One-vs-Rest ROC and AUC
        roc_data = {}
        aucs = []
        for i, cls in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            roc_data[cls] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc),
            }
        macro_auc = float(np.mean(aucs))

        eval_results[name] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_auc": macro_auc,
            "cv_macro_f1": data["cv_macro_f1"],
            "best_params": data["best_params"],
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "roc_data": roc_data,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    return eval_results


def generate_evaluation_plots(eval_results):
    """Plot and save confusion matrices, ROC curves, and comparative bar chart."""
    print("\n[Step 5/6] Generating and Saving Evaluation Charts...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix for each model + Consolidated 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(eval_results.items()):
        cm = np.array(res["confusion_matrix"])
        # Individual CM plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        plt.title(f"Confusion Matrix: {name}\nMacro-F1: {res['macro_f1']:.4f}", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("True Class", fontsize=11)
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(REPORTS_DIR / f"confusion_matrix_{safe_name}.png", dpi=300)
        plt.close()

        # Grid plot
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=axes[idx])
        axes[idx].set_title(f"{name} (Macro-F1: {res['macro_f1']:.4f})", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Class", fontsize=10)
        axes[idx].set_ylabel("True Class", fontsize=10)

    fig.suptitle("Confusion Matrix Comparison Across All Models", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    cm_all_path = REPORTS_DIR / "confusion_matrices_all.png"
    fig.savefig(cm_all_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {cm_all_path.name}")

    # 2. One-vs-Rest ROC Curves for each model & ROC Comparison
    palette = {"Legitimate": "#2ecc71", "Phishing": "#e74c3c", "Spam": "#f39c12"}

    # Individual ROC curves
    for name, res in eval_results.items():
        plt.figure(figsize=(7, 6))
        for cls in CLASS_NAMES:
            c_data = res["roc_data"][cls]
            plt.plot(c_data["fpr"], c_data["tpr"], color=palette[cls], lw=2,
                     label=f"{cls} (AUC = {c_data['auc']:.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title(f"One-vs-Rest ROC Curves: {name}\nMacro AUC = {res['macro_auc']:.4f}", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(REPORTS_DIR / f"roc_curves_{safe_name}.png", dpi=300)
        plt.close()

    # ROC Comparison (Macro AUCs)
    plt.figure(figsize=(8, 6))
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22"]
    for idx, (name, res) in enumerate(eval_results.items()):
        # Average TPR across classes for a representative macro ROC curve
        # Plot class: Phishing specifically to compare detection sensitivity
        phishing_roc = res["roc_data"]["Phishing"]
        plt.plot(phishing_roc["fpr"], phishing_roc["tpr"], color=colors[idx], lw=2,
                 label=f"{name} (Phishing AUC = {phishing_roc['auc']:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Phishing Class ROC Curve Comparison Across Models", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_comp_path = REPORTS_DIR / "roc_comparison.png"
    plt.savefig(roc_comp_path, dpi=300)
    plt.close()
    print(f"  Saved: {roc_comp_path.name}")

    # 3. Model Performance Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    comp_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": res["accuracy"],
            "Macro-F1": res["macro_f1"],
            "Phishing-F1": res["per_class"]["Phishing"]["f1_score"],
            "Macro-AUC": res["macro_auc"],
        }
        for name, res in eval_results.items()
    ])
    melted = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.ylim(0.80, 1.02)
    plt.title("Model Performance Benchmark Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("Model", fontsize=11)
    plt.legend(loc="lower right")
    plt.tight_layout()
    comp_bar_path = REPORTS_DIR / "model_comparison.png"
    plt.savefig(comp_bar_path, dpi=300)
    plt.close()
    print(f"  Saved: {comp_bar_path.name}")


def error_analysis(df, idx_test, y_test, eval_results, best_model_name):
    """Save misclassified test set examples for error analysis."""
    print("\n[Step 6/6] Conducting Error Analysis & Saving Misclassifications...")
    best_pred = eval_results[best_model_name]["y_pred"]
    best_prob = eval_results[best_model_name]["y_prob"]

    test_indices = idx_test
    errors = []

    for i, idx in enumerate(test_indices):
        true_lbl = y_test[i]
        pred_lbl = best_pred[i]
        if true_lbl != pred_lbl:
            pred_idx = CLASS_NAMES.index(pred_lbl)
            conf = float(best_prob[i][pred_idx])

            # Which other models misclassified this example?
            misclassified_by = [
                m_name for m_name, res in eval_results.items() if res["y_pred"][i] != true_lbl
            ]

            original_text = str(df.iloc[idx]["text"])
            cleaned = str(df.iloc[idx]["clean_text"])

            errors.append({
                "dataset_index": int(idx),
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "confidence": round(conf, 4),
                "misclassified_by_models": "; ".join(misclassified_by),
                "clean_snippet": cleaned[:200] + ("..." if len(cleaned) > 200 else ""),
                "original_snippet": original_text[:300].replace("\n", " ") + ("..." if len(original_text) > 300 else ""),
            })

    error_df = pd.DataFrame(errors)
    error_csv_path = REPORTS_DIR / "errors.csv"
    error_df.to_csv(error_csv_path, index=False)
    print(f"  Identified {len(error_df)} misclassified samples out of {len(y_test)} test emails ({len(error_df)/len(y_test)*100:.2f}% error rate).")
    print(f"  Saved detailed error analysis to: {error_csv_path.name}")


def save_artifacts(vectorizer, trained_models, eval_results):
    """Save models, vectorizer, and metrics JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Vectorizer
    vec_path = MODELS_DIR / "vectorizer.joblib"
    joblib.dump(vectorizer, vec_path)
    print(f"\n  Saved TF-IDF Vectorizer: {vec_path.name}")

    # 2. Save individual models
    mapping = {
        "Naive Bayes": "naive_bayes.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Linear SVM": "linear_svm.joblib",
        "Random Forest": "random_forest.joblib",
    }
    for name, filename in mapping.items():
        m_path = MODELS_DIR / filename
        joblib.dump(trained_models[name]["model"], m_path)
        print(f"  Saved {name} model -> {m_path.name}")

    # 3. Find and save Best Model by Macro-F1
    best_name = max(eval_results.keys(), key=lambda k: eval_results[k]["macro_f1"])
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(trained_models[best_name]["model"], best_path)
    print(f"  >>> Best Model: '{best_name}' (Macro-F1: {eval_results[best_name]['macro_f1']:.4f}) saved to {best_path.name} <<<")

    # 4. Save metrics.json
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "class_names": CLASS_NAMES,
        "best_model_name": best_name,
        "models": {},
    }

    for name, res in eval_results.items():
        metrics["models"][name] = {
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "macro_auc": res["macro_auc"],
            "cv_macro_f1": res["cv_macro_f1"],
            "best_params": res["best_params"],
            "per_class": res["per_class"],
            "confusion_matrix": res["confusion_matrix"],
        }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved Metrics JSON: {metrics_path.name}")

    return best_name


def print_comparison_table(eval_results, best_model_name):
    """Print clean terminal comparison table."""
    print("\n" + "=" * 90)
    print("                    MODEL PERFORMANCE BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    header = f"{'Model':<22} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Phishing-F1':<12} | {'Macro-AUC':<10} | {'5-Fold CV':<10}"
    print(header)
    print("-" * 90)

    for name, res in eval_results.items():
        marker = " [BEST]" if name == best_model_name else ""
        phishing_f1 = res["per_class"]["Phishing"]["f1_score"]
        row = (
            f"{name + marker:<22} | "
            f"{res['accuracy'] * 100:6.2f}%    | "
            f"{res['macro_f1']:8.4f}   | "
            f"{phishing_f1:10.4f}   | "
            f"{res['macro_auc']:8.4f}   | "
            f"{res['cv_macro_f1']:8.4f}"
        )
        print(row)
    print("=" * 90)


def main():
    print("=" * 70)
    print("   AI-BASED EMAIL PHISHING DETECTION: MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    df = load_and_preprocess_data()
    vectorizer, X_train, X_test, y_train, y_test, idx_test, full_df = split_and_vectorize(df)
    trained_models = build_and_tune_models(X_train, y_train)
    eval_results = evaluate_models(trained_models, X_test, y_test)
    generate_evaluation_plots(eval_results)
    best_name = save_artifacts(vectorizer, trained_models, eval_results)
    error_analysis(full_df, idx_test, y_test, eval_results, best_name)
    print_comparison_table(eval_results, best_name)
    print("\nPhase 3 Model Training and Evaluation Completed Successfully!\n")


if __name__ == "__main__":
    main()
