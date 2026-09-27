"""SHAP Explainability Module for Credit Underwriting.

Implements TreeExplainer feature attributions to satisfy regulatory
transparency, Fair Lending compliance, and banking risk oversight.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import shap


class CreditExplainer:
    """Wrapper around SHAP TreeExplainer for XGBoost Credit Default Models."""

    def __init__(self, model, feature_columns: List[str]):
        """Initialize the TreeExplainer with trained XGBoost model artifact."""
        self.model = model
        self.feature_columns = feature_columns
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, df_instance: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
        """Calculate feature-level attributions and extract the top K adverse risk factors.

        Adverse risk factors are features with positive SHAP values that push
        the prediction toward class 1 (Default).

        Args:
            df_instance: Single-row DataFrame matching trained feature column order.
            top_k: Number of adverse risk factors to return (default: 3).

        Returns:
            List of dictionaries containing feature, impact, shap_value, and explanation.
        """
        # Ensure column ordering strictly matches training schema
        df_ordered = df_instance[self.feature_columns]

        # Calculate SHAP values
        shap_vals = self.explainer.shap_values(df_ordered)

        # For XGBClassifier binary, shap_vals is 1D array of length n_features
        if isinstance(shap_vals, list):
            # Some SHAP versions return list of arrays [class_0, class_1]
            raw_vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 2:
            raw_vals = shap_vals[0]
        else:
            raw_vals = shap_vals

        feature_attributions = []
        for feature, shap_val in zip(self.feature_columns, raw_vals):
            feat_val = float(df_ordered[feature].iloc[0])
            shap_score = float(shap_val)

            # Positive SHAP value means pushing toward Default (adverse risk factor)
            impact = "adverse" if shap_score > 0 else "protective"
            description = self._generate_factor_narrative(feature, feat_val, shap_score)

            feature_attributions.append({
                "feature": feature,
                "impact": impact,
                "shap_value": round(shap_score, 4),
                "feature_value": feat_val,
                "description": description,
            })

        # Sort primarily by SHAP value descending (largest positive impact on default first)
        feature_attributions.sort(key=lambda x: x["shap_value"], reverse=True)

        # Select top K adverse factors
        return feature_attributions[:top_k]

    def _generate_factor_narrative(self, feature: str, value: float, shap_val: float) -> str:
        """Construct domain-specific financial rationale for loan officers and auditors."""
        feat_lower = feature.lower()
        is_adverse = shap_val > 0

        if "credit" in feat_lower or "score" in feat_lower:
            if is_adverse:
                return f"Credit score ({int(value)}) is below prime benchmark, significantly elevating default likelihood."
            else:
                return f"Strong credit score ({int(value)}) demonstrates healthy credit discipline, mitigating default risk."

        elif "default" in feat_lower or "history" in feat_lower:
            if is_adverse:
                return f"Borrower has {int(value)} past delinquency/default events, indicating recurring repayment risk."
            else:
                return "Zero prior default history demonstrates clean repayment track record."

        elif "loan" in feat_lower:
            if is_adverse:
                return f"Loan principal of ${value:,.2f} is substantial, contributing to debt-service strain."
            else:
                return f"Requested loan size of ${value:,.2f} represents moderate, manageable debt exposure."

        elif "income" in feat_lower:
            if is_adverse:
                return f"Annual income of ${value:,.2f} restricts debt-coverage headroom relative to loan obligations."
            else:
                return f"Robust annual income of ${value:,.2f} provides strong debt-service capacity."

        elif "age" in feat_lower:
            if is_adverse:
                return f"Age demographic ({int(value)} yrs) statistically correlates with higher financial volatility."
            else:
                return f"Age profile ({int(value)} yrs) aligns with established financial stability benchmarks."

        else:
            direction = "elevates" if is_adverse else "reduces"
            return f"Feature '{feature}' at value {value} {direction} default probability (SHAP impact: {shap_val:+.3f})."
