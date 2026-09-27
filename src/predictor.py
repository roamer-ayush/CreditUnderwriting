"""Prediction service encapsulating XGBoost inference and SHAP explainability.

Decouples machine learning inference from HTTP routing to maintain clean
architecture, unit testability, and stateless service design.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import joblib
import pandas as pd
from src.config import (
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    METADATA_PATH,
    get_risk_level,
)
from src.explainability import CreditExplainer


class UnderwritingPredictor:
    """Production inference engine for Credit Underwriting and Default Scoring."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_columns_path: Path = FEATURE_COLUMNS_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self.model_path = model_path
        self.feature_columns_path = feature_columns_path
        self.metadata_path = metadata_path

        self.model = None
        self.feature_columns: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.explainer: Optional[CreditExplainer] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model, feature definitions, and initialize SHAP explainer."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Please run 'python train_model.py' to generate model artifacts."
            )

        self.model = joblib.load(self.model_path)

        if self.feature_columns_path.exists():
            self.feature_columns = joblib.load(self.feature_columns_path)
        else:
            # Fallback to standard 5 financial underwriting features
            self.feature_columns = ["Age", "Income", "LoanAmount", "CreditScore", "DefaultHistory"]

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Initialize the SHAP explainability engine
        self.explainer = CreditExplainer(self.model, self.feature_columns)

    def prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert incoming borrower dictionary into an ordered DataFrame matching model schema."""
        # Support both snake_case and CamelCase mappings
        normalized = {
            "Age": data.get("Age", data.get("age")),
            "Income": float(data.get("Income", data.get("income"))),
            "LoanAmount": float(data.get("LoanAmount", data.get("loan_amount"))),
            "CreditScore": int(data.get("CreditScore", data.get("credit_score"))),
            "DefaultHistory": int(data.get("DefaultHistory", data.get("default_history"))),
        }
        df = pd.DataFrame([normalized])
        # Reorder to guarantee exact column ordering
        return df[self.feature_columns]

    def predict(self, borrower_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform end-to-end default scoring and SHAP feature attribution.

        Args:
            borrower_data: Dictionary or Pydantic model dump with underwriting attributes.

        Returns:
            Dictionary containing default_probability, risk_level, and top_risk_factors.
        """
        if self.model is None or self.explainer is None:
            raise RuntimeError("UnderwritingPredictor is not properly initialized with model artifacts.")

        # Step 1: Format borrower data into model DataFrame
        input_df = self.prepare_dataframe(borrower_data)

        # Step 2: Compute predicted probability for default (class 1)
        probabilities = self.model.predict_proba(input_df)
        default_prob = float(probabilities[0][1])

        # Step 3: Map continuous probability to banking risk classification tier
        risk_level = get_risk_level(default_prob)

        # Step 4: Extract top 3 adverse SHAP risk factors
        top_risk_factors = self.explainer.explain_instance(input_df, top_k=3)

        return {
            "default_probability": round(default_prob, 4),
            "risk_level": risk_level,
            "top_risk_factors": top_risk_factors,
        }
