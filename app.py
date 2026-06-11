from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ==================================================
# LOAD TRAINED PIPELINE
# ==================================================

pipeline = joblib.load("credit_pipeline.pkl")

app = FastAPI(
    title="Credit Scoring API",
    version="1.0"
)

# ==================================================
# CUSTOMER INPUT MODEL
# ==================================================

class Customer(BaseModel):

    status: str
    duration: int
    credit_history: str
    purpose: str
    amount: float
    savings: str
    employment_duration: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    present_residence: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    number_credits: int
    job: str
    people_liable: int
    telephone: str
    foreign_worker: str


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def create_features(df):

    df = df.copy()

    df["young_customer"] = (
        df["age"] < 25
    ).astype(int)

    df["long_term_loan"] = (
        df["duration"] > 24
    ).astype(int)

    median_amount = 2319.5  # German Credit median

    df["large_loan"] = (
        df["amount"] > median_amount
    ).astype(int)

    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 100],
        labels=[
            "18-25",
            "26-35",
            "36-50",
            "50+"
        ]
    )

    df["credit_duration_interaction"] = (
        df["amount"] * df["duration"]
    )

    df["amount_per_month"] = (
        df["amount"] / df["duration"]
    )

    df["credit_burden"] = (
        df["amount"] *
        df["installment_rate"]
    )

    df["senior_customer"] = (
        df["age"] >= 60
    ).astype(int)

    df["large_long_loan"] = (
        (df["large_loan"] == 1)
        &
        (df["long_term_loan"] == 1)
    ).astype(int)

    return df


# ==================================================
# CREDIT SCORING FUNCTIONS
# ==================================================

def credit_score(probability):
    return round(
        850 - (550 * probability)
    )


def risk_band(probability):

    if probability < 0.20:
        return "Low Risk"

    elif probability < 0.40:
        return "Moderate Risk"

    elif probability < 0.60:
        return "Elevated Risk"

    elif probability < 0.80:
        return "High Risk"

    return "Very High Risk"


def lending_decision(score):

    if score >= 700:
        return "Approve"

    elif score >= 600:
        return "Manual Review"

    return "Reject"


# ==================================================
# HOME ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Credit Scoring API Running"
    }


# ==================================================
# SCORING ENDPOINT
# ==================================================

@app.post("/score")
def score_customer(customer: Customer):

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    customer_df = create_features(
        customer_df
    )

    probability = pipeline.predict_proba(
        customer_df
    )[0][1]

    score = credit_score(
        probability
    )

    band = risk_band(
        probability
    )

    decision = lending_decision(
        score
    )

    return {
        "probability_of_default":
            round(float(probability), 4),

        "credit_score":
            score,

        "risk_band":
            band,

        "decision":
            decision
    }