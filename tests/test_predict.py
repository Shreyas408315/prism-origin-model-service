import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

valid_payload = {
    "rule_id": "no-unused-vars",
    "rule_family": "possible-problems",
    "severity": 2,
    "is_error": 1,
    "message_length": 46,
    "has_fix": 0,
    "fix_text_length": 0,
    "fix_range_length": 0,
    "has_suggestions": 1,
    "suggestion_count": 1,
    "changed_line_count": 4,
    "file_size_lines": 6,
    "start_line": 4,
    "finding_start_line_ratio": 0.6666666666666666,
    "finding_span_lines": 1,
    "finding_span_columns": 8,
    "pr_change_code_lines": 1,
    "pr_total_findings_in_file": 2,
    "same_rule_findings_in_file": 1,
    "same_rule_findings_in_repo": 20,
    "finding_overlaps_change": 1,
    "finding_change_distance": 0
}

def test_predict_single(client):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["decision"] in ["INTRODUCED", "PRE_EXISTING"]
    assert "threshold" in data
    assert "component_scores" in data

def test_predict_batch(client):
    response = client.post("/predict/batch", json={"items": [valid_payload, valid_payload]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert "threshold" in data
    assert "model_version" in data
    for p in data["predictions"]:
        assert "risk_score" in p

def test_predict_missing_fields(client):
    invalid_payload = {"rule_id": "no-undef"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_threshold_matches_artifact(client):
    """Verify the threshold in the API response matches the artifact's stored threshold."""
    response = client.post("/predict", json=valid_payload)
    data = response.json()
    assert abs(data["threshold"] - 0.574674670640332) < 1e-6

def test_direct_vs_api_inference(client):
    """Verify the API prediction matches direct artifact inference."""
    import joblib
    import numpy as np
    import pandas as pd

    artifact = joblib.load("model/prism_ensemble_clean.joblib")
    models = artifact["models"]
    weights = artifact["weights"]
    features = artifact["features"]

    row = pd.DataFrame([valid_payload])[features]
    probs = [weights[n] * m.predict_proba(row)[:, 1] for n, m in models.items()]
    direct_score = float(sum(probs)[0] / sum(weights.values()))

    response = client.post("/predict", json=valid_payload)
    api_score = response.json()["risk_score"]

    assert abs(direct_score - api_score) < 1e-4, f"Direct={direct_score}, API={api_score}"
