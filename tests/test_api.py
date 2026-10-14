from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_invalid_payload():
    response = client.post("/predict", json={"age": 63})  # missing required fields
    assert response.status_code == 422


def test_predict_returns_prediction_for_valid_payload(monkeypatch):
    def fake_predict_one(features):
        return {"prediction": 1, "label": "disease", "confidence": 0.87}

    monkeypatch.setattr("src.api.main.predict_one", fake_predict_one)

    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
        "ca": 0, "thal": 6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["label"] == "disease"


def test_metrics_endpoint_exposes_prometheus_format():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
