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
    assert "prism-origin" in data["model_version"]

def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["positive_class"] == "INTRODUCED"
    assert "threshold" in data
    assert data["feature_count"] == 22
