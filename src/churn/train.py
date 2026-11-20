"""Train, compare and export the churn model with full MLflow tracking.

Usage:
    python -m churn.train --config configs/train.yaml

Tracking goes to $MLFLOW_TRACKING_URI (default: local sqlite:///mlflow.db).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import skops.io as sio
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import churn
from churn.config import TrainConfig, load_config
from churn.data import download, load_dataset, sha256sum, split
from churn.evaluate import (
    best_f1_threshold,
    compute_metrics,
    feature_importance,
    save_diagnostic_plots,
)
from churn.features import INPUT_FEATURES
from churn.models import build_pipeline, expand_candidates

logger = logging.getLogger("churn.train")

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class QualityGateError(RuntimeError):
    """The champion model did not reach the minimum hold-out quality."""


def cross_validate(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int, seed: int
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Stratified K-fold CV. Returns per-fold metrics and out-of-fold probabilities."""
    oof = np.zeros(len(y), dtype=float)
    fold_metrics = []
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        fold_metrics.append(compute_metrics(y.iloc[val_idx], proba))
    return fold_metrics, oof


def summarize(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(fold_metrics)
    summary = {f"cv_{k}_mean": float(v) for k, v in frame.mean().items()}
    summary |= {f"cv_{k}_std": float(v) for k, v in frame.std(ddof=0).items()}
    return summary


def check_quality_gate(metrics: dict[str, float], config: TrainConfig) -> list[str]:
    gate = config.quality_gate
    failures = []
    if metrics["roc_auc"] < gate.min_roc_auc:
        failures.append(f"roc_auc {metrics['roc_auc']:.4f} < {gate.min_roc_auc}")
    if metrics["pr_auc"] < gate.min_pr_auc:
        failures.append(f"pr_auc {metrics['pr_auc']:.4f} < {gate.min_pr_auc}")
    return failures


def export_model(pipeline: Pipeline, metadata: dict[str, Any], out_dir: Path) -> Path:
    """Write the serving artifact: a single sklearn pipeline + JSON metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def _register(model_info, config: TrainConfig) -> str | None:
    """Register the model version and point the `champion` alias at it."""
    try:
        version = mlflow.register_model(model_info.model_uri, config.registered_model_name)
        mlflow.MlflowClient().set_registered_model_alias(
            config.registered_model_name, "champion", version.version
        )
        return str(version.version)
    except MlflowException as exc:  # e.g. a tracking backend without a registry
        logger.warning("Model registry unavailable, skipping registration: %s", exc)
        return None


def run(config: TrainConfig, config_path: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    raw_path = download(config.data.url, config.data.path, config.data.sha256)
    df = load_dataset(raw_path)
    X_train, X_test, y_train, y_test = split(df, config.data.test_size, config.seed)
    logger.info(
        "Data: %d rows, churn rate %.3f | train=%d test=%d",
        len(df),
        df["Churn"].mean(),
        len(X_train),
        len(X_test),
    )

    mlflow.set_experiment(config.experiment_name)
    # Open the experiment in MLflow UI's classic-ML view (not the GenAI one).
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    run_name = f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags({"stage": "training", "package_version": churn.__version__})
        mlflow.log_params(
            {
                "seed": config.seed,
                "cv_folds": config.cv_folds,
                "selection_metric": config.selection_metric,
                "test_size": config.data.test_size,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "data_sha256": sha256sum(raw_path),
            }
        )
        if config_path is not None:
            mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_input(
            mlflow.data.from_pandas(
                df, source=config.data.url, targets="Churn", name="telco-customer-churn"
            ),
            context="training",
        )

        # 1) Compare all candidates with cross-validation on the training split only.
        leaderboard = []
        for name, estimator, params in expand_candidates(config.candidates):
            with mlflow.start_run(run_name=name, nested=True) as child:
                pipeline = build_pipeline(estimator, params, config.seed)
                fold_metrics, oof = cross_validate(
                    pipeline, X_train, y_train, config.cv_folds, config.seed
                )
                summary = summarize(fold_metrics)
                mlflow.log_params({"estimator": estimator, **params})
                mlflow.log_metrics(summary)
                for step, fm in enumerate(fold_metrics):
                    mlflow.log_metrics({f"fold_{k}": v for k, v in fm.items()}, step=step)
                score = summary[f"cv_{config.selection_metric}_mean"]
                logger.info("%-55s cv_%s=%.4f", name, config.selection_metric, score)
                leaderboard.append(
                    {
                        "name": name,
                        "estimator": estimator,
                        "params": params,
                        "score": score,
                        "summary": summary,
                        "oof": oof,
                        "run_id": child.info.run_id,
                    }
                )

        leaderboard.sort(key=lambda r: r["score"], reverse=True)
        best = leaderboard[0]
        mlflow.MlflowClient().set_tag(best["run_id"], "champion", "true")
        logger.info(
            "Champion: %s (cv_%s=%.4f)", best["name"], config.selection_metric, best["score"]
        )

        # 2) Decision threshold from out-of-fold predictions (test set stays untouched).
        threshold = best_f1_threshold(y_train, best["oof"])

        # 3) Refit champion on the full training split and evaluate once on hold-out.
        model = build_pipeline(best["estimator"], best["params"], config.seed)
        model.fit(X_train, y_train)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba, threshold)

        mlflow.log_params({"champion": best["name"], "decision_threshold": round(threshold, 4)})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"champion_{k}": v for k, v in best["summary"].items()})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for path in save_diagnostic_plots(y_test, test_proba, threshold, tmp_dir):
                mlflow.log_artifact(str(path), artifact_path="plots")
            importance, paths = feature_importance(model, X_test, y_test, tmp_dir, config.seed)
            for path in paths:
                mlflow.log_artifact(str(path), artifact_path="plots")
            board = pd.DataFrame(
                [{"candidate": r["name"], **r["summary"]} for r in leaderboard]
            ).round(4)
            board.to_csv(tmp_dir / "leaderboard.csv", index=False)
            mlflow.log_artifact(str(tmp_dir / "leaderboard.csv"))

        # 4) Quality gate: a weak model is never registered or exported.
        failures = check_quality_gate(test_metrics, config)
        mlflow.set_tag("quality_gate", "failed" if failures else "passed")
        if failures:
            raise QualityGateError("Quality gate failed: " + "; ".join(failures))

        # 5) Log as an MLflow model (self-contained: ships the `churn` package code).
        # MLflow stores sklearn models with skops (safer than pickle). We built this
        # model in this very process, so its types are trusted by construction.
        signature = infer_signature(X_test.head(50), test_proba[:50])
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(model)),
            code_paths=[str(Path(churn.__file__).parent)],
            pip_requirements=[
                f"scikit-learn=={sklearn.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
            ],
        )
        registry_version = _register(model_info, config)

        metadata = {
            "model_name": config.registered_model_name,
            "model_version": parent.info.run_id[:8],
            "registry_version": registry_version,
            "mlflow_run_id": parent.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "candidate": best["name"],
            "estimator": best["estimator"],
            "params": best["params"],
            "threshold": threshold,
            "metrics_test": {k: round(v, 4) for k, v in test_metrics.items()},
            "metrics_cv": {k: round(v, 4) for k, v in best["summary"].items()},
            "top_features": importance.head(5)["feature"].tolist(),
            "features": INPUT_FEATURES,
            "data_sha256": sha256sum(raw_path),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "sklearn_version": sklearn.__version__,
            "package_version": churn.__version__,
        }
        export_model(model, metadata, config.output_dir)
        mlflow.log_dict(metadata, "metadata.json")

    logger.info(
        "Done in %.1fs. Test: %s | threshold=%.3f | exported to %s/",
        time.perf_counter() - started,
        ", ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()),
        threshold,
        config.output_dir,
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args(argv)

    # MLflow hint about int columns and NaN: our API validates inputs, so no NaNs reach it.
    warnings.filterwarnings("ignore", message=".*integer column", category=UserWarning)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    try:
        run(load_config(args.config), args.config)
    except QualityGateError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
