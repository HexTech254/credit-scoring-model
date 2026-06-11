import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. LOAD DATA
# ==================================================

url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"

df = pd.read_csv(url)

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ==================================================
# 2. FEATURE ENGINEERING
# ==================================================

def create_features(df):

    df = df.copy()

    # Young borrower
    df["young_customer"] = (
        df["age"] < 25
    ).astype(int)

    # Long duration loan
    df["long_term_loan"] = (
        df["duration"] > 24
    ).astype(int)

    # Large loan
    median_amount = df["amount"].median()

    df["large_loan"] = (
        df["amount"] > median_amount
    ).astype(int)

    # Age bands
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

    # Credit-duration interaction
    df["credit_duration_interaction"] = (
        df["amount"] * df["duration"]
    )

    # Monthly loan burden
    df["amount_per_month"] = (
        df["amount"] / df["duration"]
    )

    # Credit burden
    df["credit_burden"] = (
        df["amount"] *
        df["installment_rate"]
    )

    # Senior customer
    df["senior_customer"] = (
        df["age"] >= 60
    ).astype(int)

    # Large + long loan
    df["large_long_loan"] = (
        (df["large_loan"] == 1)
        &
        (df["long_term_loan"] == 1)
    ).astype(int)

    return df


df = create_features(df)

print("\nFeature Engineering Complete")

# ==================================================
# 3. TARGET AND FEATURES
# ==================================================

TARGET = "credit_risk"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==================================================
# 4. COLUMN TYPES
# ==================================================

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object", "category", "string"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumeric Features:")
print(numeric_features)

# ==================================================
# 5. TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================================
# 6. PREPROCESSOR
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# ==================================================
# 7. PIPELINE
# ==================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=5000,
            random_state=42
        )
    )
])

# ==================================================
# 8. TRAIN
# ==================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")

# ==================================================
# 9. EVALUATION
# ==================================================

y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(
    y_test,
    y_pred
))

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==================================================
# 10. CROSS VALIDATION
# ==================================================

print("\n" + "=" * 50)
print("CROSS VALIDATION")
print("=" * 50)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("ROC-AUC Scores:")
print(cv_scores)

print(
    "\nAverage ROC-AUC:",
    cv_scores.mean()
)

print(
    "Std Dev:",
    cv_scores.std()
)

# ==================================================
# 11. SAVE MODEL
# ==================================================

joblib.dump(
    pipeline,
    "credit_pipeline.pkl"
)

print("\nModel saved successfully.")
print("File: credit_pipeline.pkl")

# ==================================================
# 12. SAMPLE PREDICTION
# ==================================================

sample_customer = X_test.iloc[[0]]

probability = pipeline.predict_proba(
    sample_customer
)[0][1]

print("\nSample Prediction")
print(
    f"Probability of Default: "
    f"{probability:.4f}"
)