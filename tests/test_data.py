import pandas as pd
import pytest

from churn.data import DataValidationError, clean, download, sha256sum, split, validate


def test_validate_accepts_raw_schema(raw_df):
    validate(raw_df)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns=["tenure"]), "Missing columns"),
        (lambda d: d.head(0), "empty"),
        (lambda d: pd.concat([d, d.head(1)]), "Duplicate"),
        (lambda d: d.assign(Churn="Maybe"), "target"),
    ],
)
def test_validate_rejects_broken_data(raw_df, mutate, message):
    with pytest.raises(DataValidationError, match=message):
        validate(mutate(raw_df))


def test_clean_fixes_total_charges_and_target(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype == float
    assert not df["TotalCharges"].isna().any()
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}


def test_split_is_stratified_and_drops_id(raw_df):
    df = clean(raw_df)
    X_train, X_test, y_train, y_test = split(df, test_size=0.25, seed=0)
    assert len(X_test) == pytest.approx(0.25 * len(df), abs=1)
    assert "customerID" not in X_train.columns
    assert "Churn" not in X_train.columns
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_download_verifies_checksum(raw_csv):
    assert download("file://unused", raw_csv, sha256sum(raw_csv)) == raw_csv
    with pytest.raises(DataValidationError, match="Checksum mismatch"):
        download("file://unused", raw_csv, "0" * 64)
