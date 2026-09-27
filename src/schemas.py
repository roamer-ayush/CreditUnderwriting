"""Pydantic schemas for request validation, response serialization, and Swagger documentation.

Enforces strict input validation to guarantee model safety and clean 3NF payloads.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    """Borrower underwriting application payload validated via Pydantic."""

    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Borrower age in years (must be between 18 and 100)",
        examples=[42],
    )
    income: float = Field(
        ...,
        gt=0,
        description="Annual verified income in currency units (must be > 0)",
        examples=[65000.0],
    )
    loan_amount: float = Field(
        ...,
        gt=0,
        description="Requested loan principal amount (must be > 0)",
        examples=[350000.0],
    )
    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="Credit bureau score (FICO/CIBIL equivalent, 300 to 850)",
        examples=[580],
    )
    default_history: int = Field(
        ...,
        ge=0,
        description="Count of prior recorded loan defaults (must be >= 0)",
        examples=[2],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 42,
                "income": 65000.0,
                "loan_amount": 350000.0,
                "credit_score": 580,
                "default_history": 2,
            }
        }
    )


class RiskFactor(BaseModel):
    """Feature-level attribution detailing why a factor increased or mitigated default risk."""

    feature: str = Field(..., description="Feature name driving risk attribution")
    impact: str = Field(..., description="'adverse' (drives risk up) or 'protective' (drives risk down)")
    shap_value: float = Field(..., description="Marginal log-odds SHAP value attribution")
    feature_value: float = Field(..., description="Observed applicant feature value")
    description: str = Field(..., description="Business-readable explanation of risk influence")


class PredictionResponse(BaseModel):
    """Explainable credit decision response payload."""

    default_probability: float = Field(
        ...,
        description="Calibrated probability of default (0.000 to 1.000)",
        examples=[0.784],
    )
    risk_level: str = Field(
        ...,
        description="Underwriting risk classification tier: LOW, MEDIUM, or HIGH",
        examples=["HIGH"],
    )
    top_risk_factors: List[RiskFactor] = Field(
        ...,
        description="Top 3 adverse SHAP risk factors explaining the model outcome",
    )
    borrower_id: Optional[int] = Field(
        default=None,
        description="Database primary key assigned to the borrower record",
    )
    prediction_id: Optional[int] = Field(
        default=None,
        description="Audit log primary key assigned to this inference event",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "default_probability": 0.784,
                "risk_level": "HIGH",
                "top_risk_factors": [
                    {
                        "feature": "credit_score",
                        "impact": "adverse",
                        "shap_value": 1.452,
                        "feature_value": 580.0,
                        "description": "Credit score (580) significantly elevates default likelihood.",
                    },
                    {
                        "feature": "default_history",
                        "impact": "adverse",
                        "shap_value": 1.187,
                        "feature_value": 2.0,
                        "description": "Prior default count (2) indicates elevated delinquency recurrence.",
                    },
                    {
                        "feature": "loan_amount",
                        "impact": "adverse",
                        "shap_value": 0.742,
                        "feature_value": 350000.0,
                        "description": "Loan amount ($350,000.00) relative to income strains debt-service capacity.",
                    },
                ],
                "borrower_id": 1,
                "prediction_id": 1,
            }
        }
    )


class HealthResponse(BaseModel):
    """System health check and model operational readiness payload."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    explainer_loaded: bool
    database_connected: bool
    version: str
    timestamp: str
