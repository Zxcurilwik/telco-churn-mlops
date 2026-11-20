"""Shared fixtures. Tests run fully offline on a synthetic dataset with the
same schema as the real one, so CI never depends on a network download."""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from churn.config import TrainConfig  # noqa: E402

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_raw_dataset(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows in the raw IBM format, with a learnable churn signal."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 119, n), 2)
    total = np.round(monthly * tenure, 2)

    logit = (
        -1.2
        + 1.6 * (contract == "Month-to-month")
        + 0.8 * (internet == "Fiber optic")
        - 0.04 * tenure
        + rng.normal(0, 0.5, n)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            # Raw file stores TotalCharges as text, blank for tenure == 0.
            "TotalCharges": [" " if t == 0 else str(v) for t, v in zip(tenure, total, strict=True)],
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    for col in ADDONS:
        df[col] = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw_dataset()


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory, raw_df) -> Path:
    path = tmp_path_factory.mktemp("data") / "telco.csv"
    raw_df.to_csv(path, index=False)
    return path


def make_config(raw_csv: Path, out_dir: Path, **overrides) -> TrainConfig:
    """A tiny but complete training config: 2 candidates, 3-fold CV."""
    raw = {
        "experiment_name": "test-churn",
        "registered_model_name": "test-churn-model",
        "seed": 7,
        "data": {
            "url": "https://example.com/telco.csv",
            "sha256": None,
            "path": str(raw_csv),
            "test_size": 0.25,
        },
        "cv_folds": 3,
        "selection_metric": "roc_auc",
        "quality_gate": {"min_roc_auc": 0.6, "min_pr_auc": 0.3},
        "output_dir": str(out_dir),
        "candidates": {
            "logreg": {
                "estimator": "logistic_regression",
                "params": {"max_iter": [500]},
                "grid": {"C": [0.1, 1.0]},
            },
            "hist_gb": {
                "estimator": "hist_gradient_boosting",
                "params": {"max_iter": [30]},
            },
        },
    }
    raw.update(overrides)
    return TrainConfig.model_validate(raw)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, raw_csv):
    """Run the real training entry point once per test session (sqlite MLflow in tmp)."""
    import mlflow

    from churn.train import run

    root = tmp_path_factory.mktemp("training")
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    cwd = os.getcwd()
    os.chdir(root)  # keep MLflow's default ./mlruns artifact dir out of the repo
    try:
        metadata = run(make_config(raw_csv, root / "models"))
    finally:
        os.chdir(cwd)
    return {"root": root, "model_dir": root / "models", "metadata": metadata}
