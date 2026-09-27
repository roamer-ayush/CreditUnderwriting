"""Configuration management for Credit Underwriting & Default Scoring Engine.

Loads environment variables, configures paths, database connections,
and underwriting business rules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "credit_underwriting")

# API Server Configuration
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Model and Data Paths
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "credit_default_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_credit_data.csv"

MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "model/xgb_model.pkl")
FEATURE_COLUMNS_PATH = BASE_DIR / os.getenv("FEATURE_COLUMNS_PATH", "model/feature_columns.pkl")
METADATA_PATH = BASE_DIR / os.getenv("METADATA_PATH", "model/model_metadata.json")

# Ensure required directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

# Risk Classification Policy Rules
# Phase 21: Business Rule Layer (separate from ML model)
RISK_THRESHOLD_LOW = 0.30
RISK_THRESHOLD_MEDIUM = 0.60


def get_risk_level(probability: float) -> str:
    """Classify default probability into banking credit risk tiers.

    Business Rules:
        - 0.00 to < 0.30: LOW Risk (Standard Fast-track Approval)
        - 0.30 to < 0.60: MEDIUM Risk (Secondary Underwriting / Manual Review)
        - 0.60 to 1.00: HIGH Risk (Adverse Credit Action / Rejection)
    """
    if probability < RISK_THRESHOLD_LOW:
        return "LOW"
    elif probability < RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"
    else:
        return "HIGH"
