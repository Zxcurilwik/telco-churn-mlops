"""Loads the exported model artifact and turns validated customers into predictions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from app.schemas import EXAMPLE_CUSTOMER, Customer, Prediction
from churn.features import INPUT_FEATURES

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class ModelNotFoundError(FileNotFoundError):
    pass


class ModelService:
    def __init__(self, pipeline: Any, metadata: dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = round(float(metadata["threshold"]), 4)
        self.version = str(metadata["model_version"])

    @classmethod
    def load(cls, model_dir: Path) -> ModelService:
        model_path, meta_path = model_dir / MODEL_FILE, model_dir / METADATA_FILE
        if not model_path.exists() or not meta_path.exists():
            raise ModelNotFoundError(
                f"No model in {model_dir.resolve()}. Train one first: make train"
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        trained_with = metadata.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "Model trained with scikit-learn %s, running %s", trained_with, sklearn.__version__
            )
        # The artifact is produced by our own training pipeline (see Dockerfile),
        # never uploaded by users, so unpickling it is safe.
        service = cls(joblib.load(model_path), metadata)
        service.predict([Customer.model_validate(EXAMPLE_CUSTOMER)])  # warm-up / self-check
        logger.info("Loaded model %s (%s)", service.version, metadata.get("candidate"))
        return service

    def risk_level(self, proba: float) -> str:
        if proba >= self.threshold:
            return "high"
        if proba >= 0.6 * self.threshold:
            return "medium"
        return "low"

    def predict(self, customers: list[Customer]) -> list[Prediction]:
        frame = pd.DataFrame(
            [c.model_dump(exclude={"customerID"}) for c in customers], columns=INPUT_FEATURES
        )
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            Prediction(
                customerID=customer.customerID,
                churn_probability=round(float(p), 4),
                churn=bool(p >= self.threshold),
                risk_level=self.risk_level(float(p)),
            )
            for customer, p in zip(customers, probabilities, strict=True)
        ]
