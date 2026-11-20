"""Typed training configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    url: str
    sha256: str | None = None
    path: Path
    test_size: float = Field(0.2, gt=0, lt=1)


class QualityGate(BaseModel):
    min_roc_auc: float = 0.0
    min_pr_auc: float = 0.0


class CandidateConfig(BaseModel):
    estimator: str
    params: dict[str, list[Any]] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    experiment_name: str
    registered_model_name: str
    seed: int = 42
    data: DataConfig
    cv_folds: int = Field(5, ge=2)
    selection_metric: str = "pr_auc"
    quality_gate: QualityGate = QualityGate()
    output_dir: Path = Path("models")
    candidates: dict[str, CandidateConfig]


def load_config(path: str | Path) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig.model_validate(raw)
