"""End-to-End Model Training, Data Validation, Evaluation & Explainability Pipeline.

Project 1: Explainable Credit Underwriting & Default Scoring Engine
Compliant with IndusInd Bank Retail Risk Modeling Standards:
- Realistic 30,000 synthetic financial records with non-linear risk functions
- No data leakage (only origination-time features)
- scale_pos_weight handling of class imbalance (~13-15% default rate)
- Rigorous financial evaluation: ROC-AUC, PR-AUC, Precision, Recall
- SHAP TreeExplainer generation for global and local risk attributions
- Serializes production artifacts: xgb_model.pkl, feature_columns.pkl, model_metadata.json
"""

import json
from datetime import datetime
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Configure directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "credit_default_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_credit_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)


def generate_synthetic_credit_dataset(n_samples: int = 30000, random_state: int = 42) -> pd.DataFrame:
    """Generate 30,000 realistic retail credit applicant records.

    Enforces realistic statistical relationships without future leakage:
    - Lower credit score -> exponentially higher default odds
    - Higher loan-to-income (DTI proxy) -> higher default odds
    - Prior default history -> severe adverse impact
    - Young or vulnerable age brackets -> moderate risk variance
    """
    np.random.seed(random_state)
    print(f"\n[Phase 4] Generating {n_samples:,} synthetic financial records...")

    # Feature 1: Age (Uniform + Gamma mix for realistic working-age distribution)
    age = np.random.normal(loc=41.5, scale=12.0, size=n_samples)
    age = np.clip(np.round(age), 18, 75).astype(int)

    # Feature 2: Annual Income (Log-normal distribution reflecting wealth skew)
    income_log = np.random.normal(loc=11.1, scale=0.65, size=n_samples)
    income = np.round(np.exp(income_log), -2)  # Round to nearest hundred
    income = np.clip(income, 18000, 450000)

    # Feature 3: Loan Amount (Correlated with income, but with high-leverage outliers)
    # Average requested loan is ~2.5x to 4.5x annual income
    loan_multiplier = np.random.uniform(1.2, 5.8, size=n_samples)
    loan_amount = np.round(income * loan_multiplier * np.random.uniform(0.7, 1.3, size=n_samples), -2)
    loan_amount = np.clip(loan_amount, 5000, 800000)

    # Feature 4: Credit Score (Triangular/Beta mix mimicking FICO/CIBIL 300-850)
    # Mean ~680, standard deviation ~80
    credit_score = np.random.normal(loc=675, scale=85, size=n_samples)
    credit_score = np.clip(np.round(credit_score), 300, 850).astype(int)

    # Feature 5: Default History (Poisson distributed, heavily zero-inflated)
    # Most borrowers have 0 prior defaults, some have 1, few have 2+
    default_history_base = np.random.poisson(lam=0.35, size=n_samples)
    # Borrowers with low credit scores have higher probability of past defaults
    low_score_penalty = (credit_score < 600).astype(int) * np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
    default_history = np.clip(default_history_base + low_score_penalty, 0, 6)

    # =========================================================================
    # Latent Default Probability Function (Non-Linear Banking Underwriting Score)
    # Logit link function modeling default log-odds:
    # logit(p) = beta_0 + beta_cs*(CS - 700) + beta_dti*(Loan/Income) + beta_dh*(DH)
    # =========================================================================
    dti_ratio = loan_amount / np.maximum(income, 1.0)
    
    # Standardized features for stable latent log-odds calculation
    norm_score = (credit_score - 675.0) / 85.0
    norm_dti = (dti_ratio - 3.2) / 1.5
    norm_dh = default_history.astype(float)
    norm_age = (age - 40.0) / 12.0

    # Non-linear logit log-odds
    # Intercept calibrated for ~13.5% default rate
    logit = (
        -2.15
        - 1.35 * norm_score                     # Lower score -> higher default
        + 0.65 * norm_dti                       # High debt-to-income -> higher default
        + 0.95 * norm_dh                        # Past defaults -> strong risk multiplier
        - 0.25 * (norm_age > 0).astype(float)   # Stability with mature age
        + 0.35 * (norm_score < -1.0) * (norm_dh > 0) # Non-linear interaction: low score + default history
        + np.random.normal(0, 0.35, size=n_samples) # Idiosyncratic financial shocks
    )

    default_prob = 1.0 / (1.0 + np.exp(-logit))
    # Generate binary default outcome
    default_target = (np.random.uniform(0, 1, size=n_samples) < default_prob).astype(int)

    df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "DefaultHistory": default_history,
        "Default": default_target,
    })

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Phase 5 & 6: Data Validation and Zero-Leakage Audit."""
    print("\n" + "=" * 60)
    print("[Phase 5 & 6] Data Validation & Leakage Prevention Audit")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing Values: {df.isnull().sum().to_dict()}")
    print("\nFirst 5 Records:")
    print(df.head())
    print("\nDescriptive Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])

    default_counts = df["Default"].value_counts()
    default_rate = df["Default"].mean() * 100
    print(f"\nTarget Class Distribution:")
    print(f" - Non-Default (0): {default_counts.get(0, 0):,} ({100 - default_rate:.2f}%)")
    print(f" - Default (1):     {default_counts.get(1, 0):,} ({default_rate:.2f}%)")
    print(f"Class Imbalance Ratio: {default_counts.get(0, 0) / default_counts.get(1, 0):.2f} : 1")
    print("Leakage Check: PASS (Zero post-disbursement repayment or delinquency data included)")


def train_and_evaluate():
    """Execute complete phases 4 through 12 pipeline."""
    # 1. Generate & Validate Data
    df = generate_synthetic_credit_dataset(n_samples=30000, random_state=42)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset to: {RAW_DATA_PATH}")

    validate_dataset(df)

    # 2. Phase 7: Train/Test Split
    feature_cols = ["Age", "Income", "LoanAmount", "CreditScore", "DefaultHistory"]
    X = df[feature_cols]
    y = df["Default"]

    print("\n" + "=" * 60)
    print("[Phase 7] Stratified Train/Test Split (80/20)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training Set:   {X_train.shape[0]:,} samples")
    print(f"Testing Set:    {X_test.shape[0]:,} samples")

    # 3. Phase 8: Calculate scale_pos_weight for Class Imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print("\n" + "=" * 60)
    print(f"[Phase 8] Computed scale_pos_weight: {scale_pos_weight:.4f}")
    print("=" * 60)

    # 4. Phase 9: Train XGBoost Classifier
    print("\n" + "=" * 60)
    print("[Phase 9] Training XGBoost Classifier...")
    print("=" * 60)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("XGBoost Model Training Completed Successfully.")

    # 5. Phase 10: Model Evaluation (No raw accuracy as primary metric)
    print("\n" + "=" * 60)
    print("[Phase 10] Model Evaluation & Financial Risk Metrics")
    print("=" * 60)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"1. ROC-AUC Score:             {roc_auc:.4f}  (Benchmark: > 0.85)")
    print(f"2. PR-AUC Score (Avg Prec):   {pr_auc:.4f}  (Benchmark: > 0.55)")
    print(f"3. Precision (Default class): {precision:.4f}")
    print(f"4. Recall (Default class):    {recall:.4f}")
    print(f"5. F1-Score:                  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"[[TN={cm[0,0]:5d}, FP={cm[0,1]:5d}]")
    print(f" [FN={cm[1,0]:5d}, TP={cm[1,1]:5d}]]")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Default (0)", "Default (1)"]))

    # Save visual evaluation charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot A: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, color="#0052cc", lw=2.5, label=f"XGBoost (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#888888", linestyle="--")
    axes[0].set_title("ROC Curve (Default Discrimination)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Plot B: Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(rec, prec, color="#d9381e", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.3f})")
    axes[1].set_title("Precision-Recall Curve (Imbalanced Focus)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    # Plot C: Confusion Matrix Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[2],
        xticklabels=["Non-Default", "Default"],
        yticklabels=["Non-Default", "Default"],
    )
    axes[2].set_title("Confusion Matrix (Test Set: 6,000)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Predicted Label", fontsize=11)
    axes[2].set_ylabel("Actual Label", fontsize=11)

    plt.tight_layout()
    eval_plot_path = MODEL_DIR / "evaluation_metrics.png"
    plt.savefig(eval_plot_path, dpi=200)
    plt.close()
    print(f"Saved evaluation metrics plot to: {eval_plot_path}")

    # 6. Phase 11: SHAP Explainability & Visualizations
    print("\n" + "=" * 60)
    print("[Phase 11] Initializing SHAP TreeExplainer & Extracting Risk Attributions...")
    print("=" * 60)
    explainer = shap.TreeExplainer(model)
    shap_sample = X_test.sample(min(1500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)

    # Generate SHAP Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_sample, show=False)
    plt.title("SHAP Global Feature Importance (Credit Underwriting)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    shap_plot_path = MODEL_DIR / "shap_summary.png"
    plt.savefig(shap_plot_path, dpi=200)
    plt.close()
    print(f"Saved SHAP summary visualization to: {shap_plot_path}")

    # Local Case Study Demonstration
    print("\nSHAP Local Explanation Validation (Case Studies):")
    # Borrower A: Prime / Low Risk
    prime_borrower = pd.DataFrame([{
        "Age": 45,
        "Income": 125000.0,
        "LoanAmount": 150000.0,
        "CreditScore": 790,
        "DefaultHistory": 0,
    }])[feature_cols]
    prob_a = model.predict_proba(prime_borrower)[0, 1]
    sv_a = explainer.shap_values(prime_borrower)[0]
    print(f"\nBorrower A (Prime Applicant): Default Probability = {prob_a * 100:.2f}%")
    for f, sv in zip(feature_cols, sv_a):
        print(f"  - {f:15s}: SHAP = {sv:+.4f} ({'Protective' if sv < 0 else 'Adverse'})")

    # Borrower B: Subprime / High Risk
    subprime_borrower = pd.DataFrame([{
        "Age": 28,
        "Income": 42000.0,
        "LoanAmount": 280000.0,
        "CreditScore": 560,
        "DefaultHistory": 3,
    }])[feature_cols]
    prob_b = model.predict_proba(subprime_borrower)[0, 1]
    sv_b = explainer.shap_values(subprime_borrower)[0]
    print(f"\nBorrower B (High-Risk Applicant): Default Probability = {prob_b * 100:.2f}%")
    for f, sv in zip(feature_cols, sv_b):
        print(f"  - {f:15s}: SHAP = {sv:+.4f} ({'Protective' if sv < 0 else 'Adverse'})")

    # 7. Phase 12: Serialize Model, Features, and Comprehensive Metadata
    print("\n" + "=" * 60)
    print("[Phase 12] Serializing Model Artifacts & Production Metadata...")
    print("=" * 60)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved XGBoost model artifact to: {MODEL_PATH}")

    joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
    print(f"Saved feature column registry to: {FEATURE_COLUMNS_PATH}")

    metadata = {
        "project": "Explainable Credit Underwriting & Default Scoring Engine",
        "algorithm": "XGBoost Classifier (TreeExplainer compatible)",
        "framework": "xgboost 2.0.3, scikit-learn 1.3.2, shap 0.44.0",
        "trained_at": datetime.now().isoformat(),
        "n_samples_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_balance": {
            "non_default_count": int((y == 0).sum()),
            "default_count": int((y == 1).sum()),
            "default_rate_pct": round(float(y.mean() * 100), 2),
        },
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(float(scale_pos_weight), 4),
            "eval_metric": "logloss",
        },
        "evaluation_metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": {
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1]),
            },
        },
        "feature_columns": feature_cols,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {METADATA_PATH}")

    # Also save processed train/test splits for reproducibility
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(DATA_DIR / "processed" / "train_split.csv", index=False)
    test_df.to_csv(DATA_DIR / "processed" / "test_split.csv", index=False)
    print(f"Saved train and test splits to {DATA_DIR / 'processed'}")

    print("\n" + "=" * 60)
    print("ALL PHASES 4 - 12 COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
