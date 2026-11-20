import numpy as np

from churn.data import clean, split
from churn.features import ENGINEERED_FEATURES, add_engineered_features, build_preprocessor


def test_engineered_features(raw_df):
    df = add_engineered_features(clean(raw_df))
    assert set(ENGINEERED_FEATURES) <= set(df.columns)
    assert df["num_services"].between(0, 8).all()
    # tenure == 0 must not divide by zero
    assert np.isfinite(df["avg_monthly_charges"]).all()


def test_preprocessor_output_is_dense_and_finite(raw_df):
    X_train, X_test, *_ = split(clean(raw_df), test_size=0.25, seed=0)
    pre = build_preprocessor().fit(X_train)
    out = pre.transform(X_test)
    assert out.shape[0] == len(X_test)
    assert np.isfinite(out).all()


def test_preprocessor_ignores_unseen_categories(raw_df):
    X_train, X_test, *_ = split(clean(raw_df), test_size=0.25, seed=0)
    pre = build_preprocessor().fit(X_train)
    unseen = X_test.head(3).assign(PaymentMethod="Crypto")
    assert pre.transform(unseen).shape == pre.transform(X_test.head(3)).shape
