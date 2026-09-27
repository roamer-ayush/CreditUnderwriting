"""FastAPI Application for Explainable Credit Underwriting & Default Scoring.

Features:
- Stateless serving layer validating JSON payloads via Pydantic
- XGBoost inference with calibrated default probability
- SHAP TreeExplainer feature attributions (top 3 adverse risk factors)
- MySQL persistence for borrowers and audit predictions (3NF schema)
- Interactive OpenAPI / Swagger UI documentation at /docs
"""

from datetime import datetime
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import ENVIRONMENT
from src.database import db_manager
from src.predictor import UnderwritingPredictor
from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("credit_underwriting.api")

# Initialize FastAPI App
app = FastAPI(
    title="Explainable Credit Underwriting & Default Scoring Engine",
    description="""
### IndusInd Bank Retail Risk Modeling Preparation — Project 1

A production-grade, explainable credit decisioning microservice built with:
* **ML Core**: XGBoost Classifier trained on 30,000 retail records with `scale_pos_weight`
* **XAI (Explainable AI)**: SHAP `TreeExplainer` providing top-3 adverse risk factors
* **Validation**: Strict Pydantic models preventing invalid credit applications
* **Persistence**: MySQL 3NF relational database (`borrowers` and `predictions`)
* **Evaluation**: Focused on ROC-AUC, PR-AUC, Precision, and Recall
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predictor instance (eagerly initialized for fast request dispatching)
try:
    predictor = UnderwritingPredictor()
    logger.info("XGBoost model and SHAP TreeExplainer loaded successfully.")
except Exception as e:
    logger.warning(f"Deferred predictor initialization: {e}")
    predictor = None


@app.on_event("startup")
def startup_event():
    """Ensure model artifacts and database readiness on startup."""
    global predictor
    if predictor is None:
        try:
            predictor = UnderwritingPredictor()
            logger.info("XGBoost model and SHAP TreeExplainer loaded successfully on startup.")
        except Exception as e:
            logger.error(f"Failed to load underwriting predictor on startup: {str(e)}")


@app.get("/", tags=["General"])
def root_endpoint():
    """Root welcoming endpoint with quick links to interactive documentation."""
    return {
        "message": "Welcome to the Explainable Credit Underwriting & Default Scoring API",
        "documentation": "/docs",
        "health_check": "/health",
        "model_metrics": "/metrics",
        "endpoints": {
            "POST /predict": "Submit borrower application for scoring & top-3 SHAP adverse factor attribution",
            "GET /borrowers": "Retrieve recent underwriting decisions and audit trail",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """System health check verifying predictor readiness and database connection."""
    model_ok = predictor is not None and predictor.model is not None
    explainer_ok = predictor is not None and predictor.explainer is not None
    db_ok = db_manager._test_connection()

    overall_status = "healthy" if (model_ok and explainer_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        model_loaded=model_ok,
        explainer_loaded=explainer_ok,
        database_connected=db_ok,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/metrics", tags=["Model Analytics"])
def get_model_metrics():
    """Return trained model validation metrics, class imbalance ratio, and hyperparameters."""
    if predictor and predictor.metadata:
        return predictor.metadata
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model metadata is not available. Please ensure model training was executed.",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Underwriting"],
    summary="Evaluate Credit Application with SHAP Adverse Attributions",
)
def predict_credit_default(payload: PredictionRequest):
    """Evaluate borrower application for default risk and return top-3 SHAP adverse factors.

    - **age**: Applicant age in years (18 - 100)
    - **income**: Annual verified income in currency units (> 0)
    - **loan_amount**: Requested loan amount (> 0)
    - **credit_score**: FICO/CIBIL credit score (300 - 850)
    - **default_history**: Past recorded loan defaults (>= 0)

    Returns:
    - `default_probability`: Calibrated probability of default (0.00 to 1.00)
    - `risk_level`: Business tier ('LOW', 'MEDIUM', 'HIGH')
    - `top_risk_factors`: Top 3 adverse SHAP features driving the credit risk
    - `borrower_id` and `prediction_id`: Stored database identifiers
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not initialized. Please ensure training artifacts exist.",
        )

    borrower_dict = payload.model_dump()

    try:
        # Step 1: Run ML prediction + SHAP feature attribution
        prediction_result = predictor.predict(borrower_dict)

        # Step 2: Persist borrower profile and decision audit in MySQL (3NF schema)
        borrower_id, prediction_id = db_manager.save_borrower_and_prediction(
            borrower_data=borrower_dict,
            prediction_result=prediction_result,
        )

        # Step 3: Return explainable underwriting response
        return PredictionResponse(
            default_probability=prediction_result["default_probability"],
            risk_level=prediction_result["risk_level"],
            top_risk_factors=prediction_result["top_risk_factors"],
            borrower_id=borrower_id,
            prediction_id=prediction_id,
        )

    except Exception as e:
        logger.error(f"Inference pipeline failure: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Underwriting inference error: {str(e)}",
        )


@app.get("/borrowers", tags=["Underwriting Audit"])
def list_recent_decisions(limit: int = 10):
    """Retrieve recent underwriting decisions and audit trail from MySQL."""
    decisions = db_manager.get_recent_underwriting_decisions(limit=limit)
    return {
        "count": len(decisions),
        "limit": limit,
        "database_connected": db_manager.is_connected,
        "records": decisions,
    }


if __name__ == "__main__":
    import uvicorn
    from src.config import APP_HOST, APP_PORT

    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
