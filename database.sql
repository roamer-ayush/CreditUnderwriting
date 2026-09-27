-- =====================================================================
-- Database Schema: Explainable Credit Underwriting & Default Scoring
-- Relational 3NF Schema for Retail Banking Underwriting Engine
-- Compliant with IndusInd Bank Enterprise Architecture Guidelines
-- =====================================================================

-- Step 1: Create Database if not exists
CREATE DATABASE IF NOT EXISTS credit_underwriting
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE credit_underwriting;

-- Step 2: Drop dependent tables for clean reset (if running migrations)
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS borrowers;
SET FOREIGN_KEY_CHECKS = 1;

-- =====================================================================
-- Table 1: borrowers
-- Stores applicant demographic and financial snapshot at underwriting time.
-- Zero future leakage: only parameters known at application submission.
-- =====================================================================
CREATE TABLE IF NOT EXISTS borrowers (
    borrower_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL CHECK (age >= 18 AND age <= 100),
    income DECIMAL(12, 2) NOT NULL CHECK (income > 0),
    loan_amount DECIMAL(12, 2) NOT NULL CHECK (loan_amount > 0),
    credit_score INT NOT NULL CHECK (credit_score >= 300 AND credit_score <= 850),
    default_history INT NOT NULL DEFAULT 0 CHECK (default_history >= 0),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Performance Indexes
    INDEX idx_borrower_credit_score (credit_score),
    INDEX idx_borrower_income (income),
    INDEX idx_borrower_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- Table 2: predictions
-- Stores inference audit trail: model output, risk category, and SHAP XAI.
-- 1:N Relationship (one borrower may undergo multiple underwriting reviews).
-- =====================================================================
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    borrower_id INT NOT NULL,
    default_probability DECIMAL(5, 4) NOT NULL CHECK (default_probability >= 0.0000 AND default_probability <= 1.0000),
    risk_level ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    shap_reasons JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Foreign Key Constraint (Referential Integrity)
    CONSTRAINT fk_predictions_borrower
        FOREIGN KEY (borrower_id)
        REFERENCES borrowers(borrower_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    -- Analytical Indexes
    INDEX idx_prediction_borrower (borrower_id),
    INDEX idx_prediction_risk_level (risk_level),
    INDEX idx_prediction_prob (default_probability),
    INDEX idx_prediction_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- Sample Analytical Queries for Credit Risk Auditing
-- =====================================================================

-- Query A: Retrieve borrower risk profile with latest underwriting assessment
-- SELECT 
--     b.borrower_id,
--     b.age,
--     b.income,
--     b.loan_amount,
--     b.credit_score,
--     b.default_history,
--     p.default_probability,
--     p.risk_level,
--     p.shap_reasons,
--     p.created_at AS decision_time
-- FROM borrowers b
-- INNER JOIN predictions p ON b.borrower_id = p.borrower_id
-- ORDER BY p.created_at DESC
-- LIMIT 50;

-- Query B: Portfolio Risk Breakdown (Underwriting Funnel)
-- SELECT 
--     risk_level,
--     COUNT(*) AS total_decisions,
--     ROUND(AVG(default_probability) * 100, 2) AS avg_default_prob_pct,
--     ROUND(MIN(default_probability) * 100, 2) AS min_default_prob_pct,
--     ROUND(MAX(default_probability) * 100, 2) AS max_default_prob_pct
-- FROM predictions
-- GROUP BY risk_level;
