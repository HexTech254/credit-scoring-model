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

# What did the data teach us about credit risk?

That's where the real business value lies.

Based on our EDA, feature engineering, and coefficient analysis, the dataset suggests several insights.

# i. Credit Risk Is Not Random

The model achieved a ROC-AUC of approximately 0.80.

This means, customer characteristics contain meaningful information about future credit risk.

In other words, risky borrowers are not randomly distributed throughout the population.

# ii. Loan Duration Is an Important Risk Factor

During EDA, we found:

Longer duration → Higher risk

The model repeatedly identified duration-related variables as important.

# Business interpretation:

The longer a customer takes to repay a loan, the greater the chance that something changes in their financial situation.

Examples:

- Job loss
- Medical expenses
- Economic downturns
- Family obligations

# iii. Credit Burden Matters More Than Loan Size Alone

A large loan is not automatically risky.

What matters is, can the borrower comfortably service the debt?

That's why engineered features such as:

- amount_per_month
- credit_burden
- credit_duration_interaction

helped improve performance.

# Business interpretation:

Risk is driven by repayment pressure rather than loan amount alone.

# iv. Credit History Is Extremely Valuable

From the model's coefficient analysis, critical account / other credits elsewhere was among the strongest predictors.

This is consistent with real-world lending.

# Business interpretation:

Past financial behavior is one of the best predictors of future financial behavior.

This is one reason modern credit bureaus exist.

# v. Age Influences Risk

Our engineered feature:

- age_band

- improved the model.

This suggests, different age groups exhibit different borrowing patterns.

Possible explanations:

- Younger borrowers may have less financial stability.
- Middle-aged borrowers often have more stable income.
- Older borrowers may have different risk characteristics.

The important point is, risk varies across demographic segments.

# vi. Feature Engineering Added Real Value

One of the biggest lessons from this project.

Without engineered features:

ROC-AUC ≈ 0.76

With engineered features:

ROC-AUC ≈ 0.80

This tells us, domain knowledge can improve a model more than switching algorithms.

This is a key lesson in machine learning.

# vii. Simple Models Can Compete With Complex Models

The model compared:

- Logistic Regression
- Decision Tree
- Random Forest

Results were similar.

# Business interpretation:

A well-engineered dataset often matters more than a highly complex algorithm.

For regulated industries like banking, this is important because:

- Logistic Regression is explainable.
- Regulators can understand it.
- Decisions can be justified.

# viii. Not All Customers Should Be Treated the Same

The model effectively segments customers into different risk profiles.

Conceptually:

- Low Risk
- Moderate Risk
- High Risk
- Very High Risk

This supports:

- Risk-based pricing
- Credit limits
- Approval decisions
- Manual reviews

# Summary

Analysis of the German Credit Dataset revealed that credit risk is strongly influenced by repayment duration, credit history, and overall debt burden. Feature engineering exposed additional relationships that were not immediately visible in the raw data and improved predictive performance from approximately 0.76 ROC-AUC to 0.80 ROC-AUC. The results demonstrate that borrower behavior and financial obligations provide meaningful signals for predicting default risk, while also showing that interpretable models such as Logistic Regression can deliver strong performance when combined with thoughtful feature engineering.

