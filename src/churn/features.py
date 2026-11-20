"""Feature definitions and the preprocessing part of the model pipeline.

Everything here is pickled into the model artifact, so the API applies exactly
the same transformations as training did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ENGINEERED_FEATURES = ["num_services", "avg_monthly_charges", "charges_ratio"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features: service count and how current price compares to history."""
    out = df.copy()
    out["num_services"] = (out[SERVICE_COLUMNS] == "Yes").sum(axis=1).astype(float)
    tenure = out["tenure"].astype(float).clip(lower=1)
    out["avg_monthly_charges"] = out["TotalCharges"].astype(float) / tenure
    # > 1 means the customer now pays more than their historical average.
    out["charges_ratio"] = out["MonthlyCharges"].astype(float) / out["avg_monthly_charges"].replace(
        0, np.nan
    )
    return out


def build_preprocessor() -> Pipeline:
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    columns = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
            ("columns", columns),
        ]
    )
