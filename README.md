# Explainable Credit Underwriting & Default Scoring Engine
### Retail Risk Modeling & Regulatory-Compliant Explainable AI (XAI)
**Target: Banking Credit Risk & Machine Learning Engineering (IndusInd Bank Preparation)**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.2-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44.0-FF4081)](https://shap.readthedocs.io)
[![MySQL](https://img.shields.io/badge/MySQL-8.3-4479A1?logo=mysql&logoColor=white)](https://mysql.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


---

https://creditunderwriting.onrender.com/

---

## 1. Executive Summary & Problem Context

In retail banking and credit underwriting, deploying black-box machine learning models poses severe regulatory and operational risks. Banking authorities (such as the Reserve Bank of India - RBI and Fair Lending directives) strictly mandate that **adverse credit decisions must be accompanied by explicit, actionable, and non-discriminatory reasons**. Furthermore, financial credit default datasets are inherently imbalanced (~10–15% default rates), making raw accuracy completely deceptive.

This repository provides an end-to-end, production-grade **Explainable Credit Underwriting & Default Scoring Engine**:
1. **Machine Learning Core**: Extreme Gradient Boosting (**XGBoost**) classifier trained on 30,000 retail credit applications, mathematically tuned with `scale_pos_weight` to address class imbalance.
2. **Explainable AI (XAI)**: Integrated **SHAP TreeExplainer** extracting the **top 3 adverse risk factors** driving every credit decision in real time.
3. **Data Integrity & Zero Leakage**: Strictly leverages features known at application origination (`Age`, `Income`, `LoanAmount`, `CreditScore`, `DefaultHistory`). Post-disbursement performance metrics are strictly omitted.
4. **Stateless Microservice**: High-throughput **FastAPI** REST layer with strict **Pydantic v2** input validation.
5. **Relational Persistence**: Fully normalized **3NF MySQL schema** storing applicant demographics (`borrowers`) and audit logs (`predictions`).

---

## 2. System Architecture

```
                    ┌───────────────────────────────────┐
                    │      Client / Underwriter UI      │
                    └─────────────────┬─────────────────┘
                                      │ HTTP POST /predict
                                      ▼
                    ┌───────────────────────────────────┐
                    │           FastAPI Gateway         │
                    │   (Pydantic Schema Validation)    │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │     Prediction Service Engine     │
                    │         (src/predictor.py)        │
                    └────────┬─────────────────┬────────┘
                             │                 │
                             ▼                 ▼
                 ┌──────────────────────┐  ┌──────────────────────┐
                 │  XGBoost Classifier  │  │  SHAP TreeExplainer  │
                 │   (xgb_model.pkl)    │  │ (Top 3 Adverse XAI)  │
                 └──────────┬───────────┘  └──────────┬───────────┘
                            │                         │
                            │  Default Probability    │  Attribution Vectors
                            └────────────┬────────────┘
                                         ▼
                    ┌───────────────────────────────────┐
                    │      Business Policy Rule Tier    │
                    │    LOW (<0.30) / MED / HIGH (>=0.6)│
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │          MySQL Database           │
                    │  (borrowers ──< 1:N ──> predictions)│
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │    JSON Response (Score + XAI)    │
                    └───────────────────────────────────┘
```

---

## 3. Project Directory Structure

```
credit-underwriting/
│
├── data/
│   ├── raw/
│   │   └── credit_default_data.csv       # 30,000 synthetic banking records
│   └── processed/
│       ├── train_split.csv               # 24,000 training observations (80%)
│       └── test_split.csv                # 6,000 holdout evaluation set (20%)
│
├── model/
│   ├── xgb_model.pkl                     # Serialized XGBoost model artifact
│   ├── feature_columns.pkl               # Feature ordering definition list
│   ├── model_metadata.json               # Metrics, hyperparameters & training metadata
│   ├── evaluation_metrics.png            # ROC, PR, and Confusion Matrix charts
│   └── shap_summary.png                  # Global SHAP beeswarm importance plot
│
├── notebooks/
│   └── model_analysis.ipynb              # Comprehensive exploratory analysis & XAI
│
├── src/
│   ├── __init__.py
│   ├── config.py                         # Environment variables & business policy rules
│   ├── database.py                       # Resilient MySQL connection & 3NF persistence
│   ├── schemas.py                        # Pydantic request/response data contracts
│   ├── predictor.py                      # Decoupled inference engine
│   └── explainability.py                 # SHAP TreeExplainer attribution & narratives
│
├── train_model.py                        # End-to-end dataset generation & model training
├── main.py                               # FastAPI application & REST endpoints
├── database.sql                          # MySQL 3NF DDL schema and audit indexes
├── requirements.txt                      # Locked dependency versions
├── .env                                  # Local configuration & database credentials
├── .env.example                          # Environment configuration template
├── .gitignore                            # Secret & artifact isolation
└── README.md                             # Production documentation
```

---

## 4. 3NF Relational Database Schema (`database.sql`)

The database models retail loan underwriting adhering to **Third Normal Form (3NF)** with strict referential integrity.

```
borrowers (1) ──────< (N) predictions
```

### Table 1: `borrowers`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `borrower_id` | `INT` | `AUTO_INCREMENT, PRIMARY KEY` | Unique applicant identifier |
| `age` | `INT` | `CHECK (age >= 18 AND age <= 100)` | Applicant age at application |
| `income` | `DECIMAL(12,2)` | `CHECK (income > 0)` | Verified gross annual income |
| `loan_amount` | `DECIMAL(12,2)` | `CHECK (loan_amount > 0)` | Requested loan principal |
| `credit_score` | `INT` | `CHECK (credit_score BETWEEN 300 AND 850)` | FICO/CIBIL credit bureau score |
| `default_history`| `INT` | `DEFAULT 0, CHECK (default_history >= 0)` | Number of past loan defaults |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Record timestamp |

### Table 2: `predictions`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `prediction_id` | `INT` | `AUTO_INCREMENT, PRIMARY KEY` | Audit log primary key |
| `borrower_id` | `INT` | `FOREIGN KEY REFERENCES borrowers(borrower_id)` | Linked applicant ID |
| `default_probability` | `DECIMAL(5,4)` | `CHECK (default_probability BETWEEN 0 AND 1)`| Model default probability |
| `risk_level` | `ENUM` | `'LOW', 'MEDIUM', 'HIGH'` | Policy risk classification |
| `shap_reasons` | `JSON` | `NOT NULL` | Serialized top 3 SHAP adverse factors |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Decision execution timestamp |

---

## 5. Machine Learning Methodology

### 5.1 Dataset Generation & Zero Leakage Audit
A 30,000-record synthetic banking dataset was generated via non-linear economic logit interactions:
- **Credit Score**: Strong inverse relationship with default probability.
- **Debt Burden**: High Loan-to-Income ratio compounds probability of default.
- **Delinquency Memory**: Prior default history acts as an exponential risk multiplier.
- **Leakage Elimination**: Zero post-disbursement features (e.g., total payments made, days delinquent) are included, avoiding lookahead bias.

### 5.2 Class Imbalance Handling
Financial default rates average 13–15%. To prevent the model from trivial non-default classification, the training pipeline computes:
$$\text{scale\_pos\_weight} = \frac{\sum (y_{\text{train}} == 0)}{\sum (y_{\text{train}} == 1)} \approx 6.4$$

### 5.3 Financial Evaluation Benchmarks
In credit risk, False Negatives (granting a loan to a borrower who defaults) cost up to 10x more than False Positives (rejecting a safe applicant). Evaluation centers on discrimination:
* **ROC-AUC Score**: $\approx 0.88 - 0.91$ (Strong borrower discrimination across cutoffs)
* **PR-AUC Score**: $\approx 0.58 - 0.65$ (High precision across true default recalls)
* **Recall (Default class)**: $\approx 80\% - 85\%$ (Capturing high-risk applicants)

---

## 6. Explainable AI (SHAP TreeExplainer)

For every loan application, the service uses `shap.TreeExplainer` to compute exact log-odds attributions:
$$g(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

Where $\phi_i > 0$ represents an **adverse risk factor** (pushing the applicant toward default).

### Case Study Comparison:

#### Case A: Prime Borrower (Fast-Track Approval)
* **Input**: Age 45, Income $125,000, Loan $150,000, Credit Score 790, Default History 0.
* **Result**: Default Probability: `4.2%` | Risk Tier: **`LOW`**
* **Attribution**: High credit score and zero default history heavily decrease risk ($\phi < 0$).

#### Case B: Subprime Borrower (Adverse Action Notice)
* **Input**: Age 28, Income $42,000, Loan $280,000, Credit Score 560, Default History 3.
* **Result**: Default Probability: `84.6%` | Risk Tier: **`HIGH`**
* **Top 3 Adverse Factors**:
  1. `Credit Score` ($\phi = +1.48$): Score of 560 is below prime benchmark.
  2. `Default History` ($\phi = +1.21$): 3 prior defaults demonstrate elevated delinquency risk.
  3. `Loan Amount` ($\phi = +0.72$): $280,000 loan strains debt-service capacity.

---

## 7. Step-by-Step Installation & Run Guide

### Step 1: Clone & Setup Virtual Environment
```powershell
# Navigate to directory
cd credit-underwriting

# Create Python 3.11 virtual environment
python -m venv venv

# Activate venv on Windows:
.\venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

Verify installation:
```powershell
python -c "import xgboost, shap, fastapi, pandas, sklearn; print('All imports successful!')"
```

### Step 3: Configure Database & Environment
Copy the template configuration:
```powershell
cp .env.example .env
```
Update `.env` with your local MySQL credentials:
```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=credit_underwriting
```

Initialize the database schema:
```powershell
mysql -u root -p < database.sql
```

### Step 4: Train Model & Generate Artifacts
Run the comprehensive training and explainability pipeline:
```powershell
python train_model.py
```
This script will:
1. Synthesize 30,000 realistic records into `data/raw/credit_default_data.csv`.
2. Perform stratified 80/20 train/test split.
3. Compute `scale_pos_weight` and train `XGBClassifier`.
4. Generate evaluation metrics, ROC curve, and PR curve in `model/evaluation_metrics.png`.
5. Run `shap.TreeExplainer` and save global feature importance to `model/shap_summary.png`.
6. Serialize `model/xgb_model.pkl`, `model/feature_columns.pkl`, and `model/model_metadata.json`.

### Step 5: Start the FastAPI Underwriting Server & Web Dashboard
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

1. **Interactive Web Dashboard**: Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser.
   - Live interactive sliders and borrower archetype presets (Prime, Borderline, Subprime).
   - Real-time animated radial risk gauge and calibrated default probability.
   - Interactive SHAP feature attribution bars with domain regulatory narratives.
   - Real-time underwriting audit trail and model analytics modal with evaluation curves.
2. **OpenAPI / Swagger UI Documentation**: Available at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

---

## 8. API Documentation & Interactive Examples

### Endpoint: `POST /predict`
Evaluates a loan applicant and returns the calibrated probability and top 3 adverse SHAP factors.

#### Request Example:
```json
POST /predict
Content-Type: application/json

{
  "age": 42,
  "income": 65000.0,
  "loan_amount": 350000.0,
  "credit_score": 580,
  "default_history": 2
}
```

#### Response Example:
```json
{
  "default_probability": 0.7842,
  "risk_level": "HIGH",
  "top_risk_factors": [
    {
      "feature": "CreditScore",
      "impact": "adverse",
      "shap_value": 1.4521,
      "feature_value": 580.0,
      "description": "Credit score (580) is below prime benchmark, significantly elevating default likelihood."
    },
    {
      "feature": "DefaultHistory",
      "impact": "adverse",
      "shap_value": 1.1865,
      "feature_value": 2.0,
      "description": "Borrower has 2 past delinquency/default events, indicating recurring repayment risk."
    },
    {
      "feature": "LoanAmount",
      "impact": "adverse",
      "shap_value": 0.7419,
      "feature_value": 350000.0,
      "description": "Loan principal of $350,000.00 is substantial, contributing to debt-service strain."
    }
  ],
  "borrower_id": 1,
  "prediction_id": 1
}
```

### Testing Invalid Input (Validation Enforcement)
```json
POST /predict
{
  "age": 15,
  "income": -5000,
  "loan_amount": 0,
  "credit_score": 950,
  "default_history": -1
}
```
**Response: `422 Unprocessable Entity`**
```json
{
  "detail": [
    {"loc": ["body", "age"], "msg": "Input should be greater than or equal to 18"},
    {"loc": ["body", "income"], "msg": "Input should be greater than 0"},
    {"loc": ["body", "credit_score"], "msg": "Input should be less than or equal to 850"}
  ]
}
```

---

## 9. Production Deployment Guide

This project is fully containerized and production-ready across multiple deployment architectures.

### Option A: Docker Compose (Full Stack with MySQL 8.0)
Spins up the FastAPI application and a dedicated MySQL 8 container with automatic schema provisioning:
```powershell
docker compose up --build -d
```
* **Web Service**: Accessible at `http://localhost:8000`
* **MySQL Database**: Running on `localhost:3306` with persisted volume `mysql_data`
* **Health Checks**: Automated container status polling via `/health`

### Option B: Cloud Containers (Google Cloud Run / AWS ECS / Azure Container Apps)
1. Build and push the Docker image:
```bash
docker build -t gcr.io/<YOUR-PROJECT>/credit-underwriting:latest .
docker push gcr.io/<YOUR-PROJECT>/credit-underwriting:latest
```
2. Deploy as a managed serverless container:
```bash
gcloud run deploy credit-underwriting \
  --image gcr.io/<YOUR-PROJECT>/credit-underwriting:latest \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars DB_HOST=<CLOUD_SQL_IP>,DB_USER=<USER>,DB_PASSWORD=<PASS>,DB_NAME=credit_underwriting
```

### Option C: Platform as a Service (Render / Railway)
* **Render**: Pre-configured via [`render.yaml`](render.yaml) blueprint. Link your repository, and Render automatically executes dependency installation, runs `train_model.py` to bake artifacts, and launches Uvicorn.
* **Railway / Heroku**: Utilizes the included [`Procfile`](Procfile):
  ```
  web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
  ```

---

## 10. Verification & Completion Checklist

| Phase | Milestone | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Phase 1-3** | Setup, venv, requirements, MySQL schema | Verified | Locked dependencies, 3NF DDL |
| **Phase 4-6** | Dataset Generation & Leakage Audit | Verified | 30,000 records, origination features only |
| **Phase 7-9** | Stratified Split, Imbalance, XGBoost | Verified | `scale_pos_weight`, max_depth=6 |
| **Phase 10** | Quantitative Evaluation | Verified | ROC-AUC, PR-AUC, Confusion Matrix |
| **Phase 11-12**| SHAP TreeExplainer & Serialization | Verified | Top 3 adverse reasons, artifacts saved |
| **Phase 13-16**| Pydantic Schema, Predictor & FastAPI | Verified | Stateless API, `/predict` endpoint |
| **Phase 17** | MySQL Integration | Verified | Atomicity, audit logging, resilient fallback |
| **Phase 18-20**| Swagger & Adverse Explanation Testing| Verified | Tested edge cases and validation |
| **Phase 21-25**| Policy Tiering & Architecture Docs | Verified | Interactive docs, Swagger, README |
