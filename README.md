# Credit Risk Scoring System

## Overview

This project builds an end-to-end machine learning system for credit risk prediction using the German Credit Dataset. The objective is to predict whether a loan applicant is likely to become a risky customer and assist lending institutions in making informed credit decisions.

The project covers the complete machine learning lifecycle:

- Data Exploration
- Feature Engineering
- Model Development
- Model Evaluation
- Cross-Validation
- Production Pipeline Creation
- FastAPI Deployment

---

## Business Problem

Financial institutions face significant losses from loan defaults. Before approving a loan, lenders need a reliable method to estimate the probability that a customer will fail to repay their debt.

The goal of this project is to develop a predictive credit scoring model capable of:

- Predicting risky borrowers
- Estimating probability of default
- Assigning credit scores
- Categorizing customers into risk bands
- Supporting lending decisions

---

## Dataset

German Credit Dataset

Source:
https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv

Dataset Characteristics:

- 1,000 customers
- 20 original features
- Financial and demographic information
- Binary target variable

Target Variable:

- 0 = Safe Customer
- 1 = Risky Customer

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

### Default Rate Analysis

Measured the percentage of risky customers within the dataset.

### Loan Amount Analysis

Compared average loan amounts between safe and risky borrowers.

### Loan Duration Analysis

Investigated the relationship between loan duration and credit risk.

### Customer Demographics

Explored age distributions and housing status.

### Feature Relationships

Examined how variables such as:

- Loan Amount
- Duration
- Installment Rate
- Credit History

influence credit risk.

### Key Findings

- Longer loan durations are associated with higher risk.
- Customers with larger credit burdens show increased default probability.
- Credit history is one of the strongest predictors.
- Risk patterns vary across age groups.

---

## Feature Engineering

Several domain-inspired features were created to improve predictive performance.

### Young Customer

Identifies customers younger than 25.

```python
young_customer = age < 25
