"""Dataset download, validation, cleaning and splitting."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

ID_COLUMN = "customerID"
TARGET = "Churn"
RAW_COLUMNS = [ID_COLUMN, *CATEGORICAL_FEATURES, *NUMERIC_FEATURES, TARGET]


class DataValidationError(ValueError):
    """Raised when the raw dataset does not match the expected contract."""


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, sha256: str | None = None) -> Path:
    """Download the dataset once; verify its checksum when one is configured."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", url, path)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - URL comes from our config
        tmp.replace(path)
    if sha256:
        actual = sha256sum(path)
        if actual != sha256:
            raise DataValidationError(
                f"Checksum mismatch for {path}: expected {sha256}, got {actual}. "
                "Delete the file to re-download, or update data.sha256 in the config."
            )
    return path


def validate(df: pd.DataFrame) -> None:
    """Fail fast on schema problems instead of training on broken data."""
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if df.empty:
        raise DataValidationError("Dataset is empty")
    if df[ID_COLUMN].duplicated().any():
        raise DataValidationError("Duplicate customer IDs found")
    bad_target = set(df[TARGET].dropna().unique()) - {"Yes", "No"}
    if bad_target or df[TARGET].isna().any():
        raise DataValidationError(f"Unexpected target values: {bad_target or 'NaN'}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known quirks of the raw IBM file.

    TotalCharges is stored as text and is blank for brand-new customers
    (tenure == 0); they have not been billed yet, so the right value is 0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", "0"), errors="coerce"
    ).astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return clean(df)


def split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = df.drop(columns=[ID_COLUMN, TARGET])
    target = df[TARGET]
    return train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=seed
    )
