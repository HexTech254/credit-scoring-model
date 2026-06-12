# Credit Risk Scoring Project Report

## Executive Summary

This project developed a machine learning-based credit scoring system capable of predicting customer credit risk using demographic and financial information.

The final model achieved approximately 0.80 ROC-AUC and was packaged into a deployable FastAPI service.

---

# Business Problem

Financial institutions must balance profitability with risk management.

Approving high-risk borrowers increases default losses while rejecting low-risk borrowers reduces potential revenue.

This project aims to support credit decisions through predictive modeling.

---

# Dataset Overview

Dataset: German Credit Dataset

Observations: 1,000

Target Variable:

- 0 = Safe Customer
- 1 = Risky Customer

Features:

- Credit History
- Loan Amount
- Loan Duration
- Savings Status
- Employment Duration
- Housing
- Age
- Installment Rate

---

# Exploratory Data Analysis

Key findings:

- Risk increases with longer loan duration.
- Credit history strongly influences risk.
- Larger financial obligations correlate with higher default probability.
- Age-based risk patterns exist.

---

# Feature Engineering

Created:

- young_customer
- long_term_loan
- large_loan
- age_band
- amount_per_month
- credit_burden
- large_long_loan

Feature engineering improved model performance significantly.

---

# Model Development

Models evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Evaluation Metric:

ROC-AUC

---

# Results

| Model | ROC-AUC |
|---------|---------:|
| Logistic Regression | 0.80 |
| Decision Tree | 0.76 |
| Random Forest | 0.80 |

---

# Cross Validation

5-Fold Stratified Cross Validation

Average ROC-AUC ≈ 0.79

The model demonstrated stable performance across multiple folds.

---

# Deployment Architecture

User Request
      ↓
FastAPI
      ↓
Feature Engineering
      ↓
Preprocessing Pipeline
      ↓
Logistic Regression
      ↓
Probability of Default
      ↓
Credit Score
      ↓
Decision

---

# Conclusion

The project successfully demonstrates the complete machine learning lifecycle, from exploratory analysis and feature engineering to model deployment.

The final system can be integrated into lending workflows to support data-driven credit decisions.
