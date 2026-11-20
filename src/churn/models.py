"""Candidate estimators and the full (preprocessing + model) pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from churn.config import CandidateConfig
from churn.features import build_preprocessor

ESTIMATORS: dict[str, type[ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def build_pipeline(estimator: str, params: dict[str, Any], seed: int) -> Pipeline:
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}; choose from {sorted(ESTIMATORS)}")
    model = ESTIMATORS[estimator](random_state=seed, **params)
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def expand_candidates(
    candidates: dict[str, CandidateConfig],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (run_name, estimator, params) for every grid point of every candidate."""
    for name, cand in candidates.items():
        for point in ParameterGrid({**cand.params, **cand.grid}):
            tuned = {k: point[k] for k in cand.grid}
            suffix = ",".join(f"{k}={v}" for k, v in tuned.items())
            yield (f"{name}[{suffix}]" if suffix else name), cand.estimator, dict(point)
