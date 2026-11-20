import pytest
from fastapi.testclient import TestClient

from app.schemas import EXAMPLE_CUSTOMER, MAX_BATCH_SIZE


@pytest.fixture(scope="module")
def client(trained, monkeypatch_module):
    from app.settings import get_settings

    monkeypatch_module.setenv("MODEL_DIR", str(trained["model_dir"]))
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as test_client:  # runs the lifespan (model loading)
        yield test_client
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_health(client, trained):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": trained["metadata"]["model_version"],
    }


def test_model_info(client):
    info = client.get("/model/info").json()
    assert {"threshold", "metrics_test", "candidate", "mlflow_run_id"} <= info.keys()


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn" in response.text


def test_predict(client):
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn"] == (body["churn_probability"] >= body["threshold"])
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["customerID"] == EXAMPLE_CUSTOMER["customerID"]
    assert "X-Request-ID" in response.headers


def test_predict_batch(client):
    loyal = EXAMPLE_CUSTOMER | {
        "customerID": "B",
        "Contract": "Two year",
        "tenure": 70,
        "TotalCharges": 2000.0,
    }
    response = client.post("/predict/batch", json={"customers": [EXAMPLE_CUSTOMER, loyal]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [p["customerID"] for p in predictions] == [EXAMPLE_CUSTOMER["customerID"], "B"]
    # Short month-to-month customer is riskier than a long two-year one.
    assert predictions[0]["churn_probability"] > predictions[1]["churn_probability"]


@pytest.mark.parametrize(
    "patch",
    [
        {"tenure": -1},
        {"Contract": "Weekly"},
        {"SeniorCitizen": 2},
        {"MonthlyCharges": "a lot"},
        {"unexpected_field": 1},
        {"PhoneService": "No", "MultipleLines": "Yes"},
        {"InternetService": "No"},  # add-ons still say Yes/No -> inconsistent
    ],
)
def test_predict_rejects_invalid_input(client, patch):
    assert client.post("/predict", json=EXAMPLE_CUSTOMER | patch).status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    too_many = {"customers": [EXAMPLE_CUSTOMER] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_startup_fails_without_model(tmp_path, monkeypatch):
    from app.model_service import ModelNotFoundError, ModelService

    with pytest.raises(ModelNotFoundError):
        ModelService.load(tmp_path)
