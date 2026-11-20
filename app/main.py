"""HTTP API for the churn model.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse

import churn
from app.model_service import ModelService
from app.schemas import BatchRequest, BatchResponse, Customer, Health, PredictionResponse
from app.settings import get_settings

logger = logging.getLogger("app")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Fail fast: a container without a working model must not report healthy.
    app.state.model = ModelService.load(settings.model_dir)
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    version=churn.__version__,
    description=(
        "Predicts the probability that a telecom customer will leave. "
        "Model: scikit-learn pipeline trained on the IBM Telco Customer Churn "
        "dataset, experiments tracked in MLflow."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    if request.url.path != "/health":
        logger.info(
            "%s %s %s %.1fms id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
    return response


def get_model(request: Request) -> ModelService:
    return request.app.state.model


Model = Annotated[ModelService, Depends(get_model)]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=Health, tags=["service"])
def health(model: Model) -> Health:
    return Health(status="ok", model_version=model.version)


@app.get("/model/info", tags=["service"])
def model_info(model: Model) -> dict[str, Any]:
    """Training metadata: champion model, params, hold-out metrics, threshold."""
    return model.metadata


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: Customer, model: Model) -> PredictionResponse:
    prediction = model.predict([customer])[0]
    return PredictionResponse(
        **prediction.model_dump(), threshold=model.threshold, model_version=model.version
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, model: Model) -> BatchResponse:
    return BatchResponse(
        threshold=model.threshold,
        model_version=model.version,
        predictions=model.predict(batch.customers),
    )
