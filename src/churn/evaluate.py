"""Metrics, decision-threshold selection and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI, Docker build

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1. Chosen on out-of-fold predictions, never on test."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last precision/recall pair has no threshold attached.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def save_diagnostic_plots(y_true, proba, threshold: float, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title("ROC curve (test)")
    paths.append(_save(fig, out_dir / "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax, name="model")
    ax.axhline(float(np.mean(y_true)), ls="--", color="grey", lw=1, label="baseline")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend()
    paths.append(_save(fig, out_dir / "pr_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        (np.asarray(proba) >= threshold).astype(int),
        display_labels=["stay", "churn"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion matrix @ threshold={threshold:.2f}")
    paths.append(_save(fig, out_dir / "confusion_matrix.png"))
    return paths


def feature_importance(
    pipeline: Pipeline, X: pd.DataFrame, y, out_dir: Path, seed: int
) -> tuple[pd.DataFrame, list[Path]]:
    """Permutation importance on raw input columns (model-agnostic, human-readable)."""
    result = permutation_importance(
        pipeline, X, y, scoring="average_precision", n_repeats=5, random_state=seed
    )
    table = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_importance.csv"
    table.to_csv(csv_path, index=False)

    top = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3b6ea5")
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    ax.set_title("Permutation feature importance (test)")
    return table, [csv_path, _save(fig, out_dir / "feature_importance.png")]


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
