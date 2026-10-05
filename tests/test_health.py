import pytest
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "eslint-surface-hybrid-ensemble-v1"

def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["positive_class"] == "surface"
    assert data["negative_class"] == "suppressed"
    assert data["threshold"] == pytest.approx(0.5050000000000001)
    assert data["feature_count"] == 55
    assert data["input_feature_count"] == 43
    assert data["model_family"] == "hybrid_ensemble"
    assert data["includes_message_tfidf"] is True
