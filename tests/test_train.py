import json

import mlflow
import numpy as np
import pytest

from churn.evaluate import best_f1_threshold, compute_metrics
from churn.models import build_pipeline, expand_candidates
from churn.train import QualityGateError, run
from tests.conftest import make_config


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.6, 0.55, 0.8, 0.9])
    metrics = compute_metrics(y, proba, threshold=0.5)
    assert {"roc_auc", "pr_auc", "f1", "recall", "brier"} <= metrics.keys()
    threshold = best_f1_threshold(y, proba)
    assert compute_metrics(y, proba, threshold)["f1"] >= metrics["f1"]


def test_expand_candidates_builds_grid(tmp_path, raw_csv):
    config = make_config(raw_csv, tmp_path)
    names = [name for name, *_ in expand_candidates(config.candidates)]
    assert names == ["logreg[C=0.1]", "logreg[C=1.0]", "hist_gb"]


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="Unknown estimator"):
        build_pipeline("svm", {}, seed=0)


def test_training_exports_artifacts(trained):
    model_dir, metadata = trained["model_dir"], trained["metadata"]
    assert (model_dir / "model.joblib").exists()
    saved = json.loads((model_dir / "metadata.json").read_text())
    assert saved == metadata
    assert 0 < saved["threshold"] < 1
    assert saved["metrics_test"]["roc_auc"] >= 0.6


def test_training_is_tracked_in_mlflow(trained):
    run_id = trained["metadata"]["mlflow_run_id"]
    parent = mlflow.get_run(run_id)
    assert parent.data.tags["quality_gate"] == "passed"
    assert "test_roc_auc" in parent.data.metrics
    children = mlflow.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
    )
    assert len(children) == 3  # one nested run per grid point
    assert trained["metadata"]["registry_version"] is not None


def test_quality_gate_blocks_weak_model(tmp_path, raw_csv, trained):
    config = make_config(
        raw_csv, tmp_path / "models", quality_gate={"min_roc_auc": 0.999, "min_pr_auc": 0}
    )
    with pytest.raises(QualityGateError):
        run(config)
    assert not (tmp_path / "models" / "model.joblib").exists()
