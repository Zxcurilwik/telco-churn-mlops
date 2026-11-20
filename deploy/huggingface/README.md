---
title: Telco Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Customer churn prediction API (FastAPI + MLflow)
---

# Telco Customer Churn Predictor

FastAPI service that predicts the probability that a telecom customer will churn.
The Docker build downloads the IBM Telco Customer Churn dataset, trains and
selects the model (experiments tracked with MLflow), checks a quality gate and
serves the result.

- Demo UI: `/`
- Swagger: `/docs`
- Model metadata: `/model/info`

This Space is deployed automatically from GitHub Actions after CI passes.
