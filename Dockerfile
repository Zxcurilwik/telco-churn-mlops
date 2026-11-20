# syntax=docker/dockerfile:1
#
# Multi-stage build:
#   trainer  — training toolchain (MLflow etc.); also used by docker-compose
#   model    — downloads data, trains, runs the quality gate (build fails if it fails)
#   runtime  — slim serving image: FastAPI + the trained artifact only (default target)
#
# The same Dockerfile is used locally, in CI, on Hugging Face Spaces and on Render.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet
WORKDIR /app

# ---------------------------------------------------------------------------
FROM base AS trainer
COPY requirements.txt requirements-train.txt ./
RUN pip install -r requirements-train.txt
COPY configs ./configs
COPY src ./src

# ---------------------------------------------------------------------------
FROM trainer AS model
RUN python -m churn.train --config configs/train.yaml

# ---------------------------------------------------------------------------
FROM base AS runtime
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 app
COPY --chown=app src ./src
COPY --chown=app app ./app
COPY --from=model --chown=app /app/models ./models
USER app

ENV MODEL_DIR=/app/models \
    PORT=7860 \
    WEB_CONCURRENCY=1
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# PORT is overridden by platforms that inject their own (e.g. Render).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
