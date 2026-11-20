PY ?= python
export PYTHONPATH := src:.
export MLFLOW_DISABLE_AGENT_HINT := 1
export GIT_PYTHON_REFRESH := quiet

.DEFAULT_GOAL := help
.PHONY: help install train mlflow-ui serve test lint format docker-build docker-run smoke up down clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install: ## Install dev dependencies + the package (editable)
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

train: ## Download data, run experiments (MLflow), export best model to models/
	$(PY) -m churn.train --config configs/train.yaml

mlflow-ui: ## Open MLflow UI for local runs at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve: ## Run the API with auto-reload at http://localhost:8000
	uvicorn app.main:app --reload --port 8000

test: ## Run tests with coverage
	pytest --cov --cov-report=term-missing

lint: ## Lint and check formatting
	ruff check .
	ruff format --check .

format: ## Auto-fix lint issues and format code
	ruff check --fix .
	ruff format .

docker-build: ## Build the production image (trains the model inside)
	docker build -t telco-churn-api .

docker-run: ## Run the production image at http://localhost:8000
	docker run --rm -p 8000:7860 telco-churn-api

smoke: ## Smoke-test a running API (URL=http://localhost:8000)
	$(PY) scripts/smoke_test.py $(or $(URL),http://localhost:8000)

up: ## MLflow server + training + API via docker compose
	docker compose up --build

down: ## Stop docker compose stack
	docker compose down

clean: ## Remove local artifacts
	rm -rf mlruns mlflow.db models/*.joblib models/*.json .pytest_cache .ruff_cache .coverage
