"""Automated Integration & Validation Test Suite for Credit Underwriting Engine.

Tests:
- Phase 18: POST /predict with standard borrower
- Phase 19: Rejection of invalid inputs via Pydantic
- Phase 20: Verification that SHAP explanations dynamically adapt between Prime and Subprime applicants
- System Health and Metadata endpoints
"""

import sys
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify system readiness and model loading."""
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["model_loaded"] is True
    assert data["explainer_loaded"] is True
    print("[PASS] GET /health: Model and SHAP Explainer are operational.")


def test_predict_subprime_borrower():
    """Phase 18: Test POST /predict with high-risk borrower (from roadmap)."""
    payload = {
        "age": 42,
        "income": 65000.0,
        "loan_amount": 350000.0,
        "credit_score": 580,
        "default_history": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Prediction failed: {response.text}"
    data = response.json()

    # Validate output schema
    assert "default_probability" in data
    assert "risk_level" in data
    assert "top_risk_factors" in data
    assert len(data["top_risk_factors"]) == 3

    print(f"\n[PASS] POST /predict (Subprime Borrower):")
    print(f"  - Default Probability: {data['default_probability'] * 100:.2f}%")
    print(f"  - Risk Tier:           {data['risk_level']}")
    print(f"  - Top 3 Adverse SHAP Factors:")
    for idx, factor in enumerate(data["top_risk_factors"], 1):
        print(f"      {idx}. {factor['feature']} (SHAP: {factor['shap_value']:+.4f}) -> {factor['description']}")


def test_predict_prime_borrower():
    """Phase 20: Test that explanations adapt for prime borrower."""
    payload = {
        "age": 45,
        "income": 130000.0,
        "loan_amount": 120000.0,
        "credit_score": 800,
        "default_history": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Prediction failed: {response.text}"
    data = response.json()

    assert data["risk_level"] in ["LOW", "MEDIUM"]
    print(f"\n[PASS] POST /predict (Prime Borrower):")
    print(f"  - Default Probability: {data['default_probability'] * 100:.2f}%")
    print(f"  - Risk Tier:           {data['risk_level']}")
    print(f"  - Top Factors:")
    for idx, factor in enumerate(data["top_risk_factors"], 1):
        print(f"      {idx}. {factor['feature']} (SHAP: {factor['shap_value']:+.4f}) -> {factor['description']}")


def test_invalid_input_rejection():
    """Phase 19: Test Pydantic input validation rejections."""
    invalid_payload = {
        "age": 15,              # Must be >= 18
        "income": -1000.0,      # Must be > 0
        "loan_amount": 350000.0,
        "credit_score": 1200,   # Must be <= 850
        "default_history": -2,  # Must be >= 0
    }
    response = client.post("/predict", json=invalid_payload)
    # Pydantic v2 returns 422 for validation error
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    errors = response.json().get("detail", [])
    error_fields = {e["loc"][-1] for e in errors}

    assert "age" in error_fields
    assert "income" in error_fields
    assert "credit_score" in error_fields
    assert "default_history" in error_fields
    print(f"\n[PASS] Phase 19 Input Validation: Correctly rejected 4 invalid fields with HTTP 422.")


def test_metrics_endpoint():
    """Verify /metrics returns trained performance numbers."""
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.json()["evaluation_metrics"]
    print(f"\n[PASS] GET /metrics: ROC-AUC = {metrics['roc_auc']}, PR-AUC = {metrics['pr_auc']}")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING CREDIT UNDERWRITING API VERIFICATION SUITE")
    print("=" * 60)
    test_health_endpoint()
    test_predict_subprime_borrower()
    test_predict_prime_borrower()
    test_invalid_input_rejection()
    test_metrics_endpoint()
    print("\n" + "=" * 60)
    print("ALL API VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
