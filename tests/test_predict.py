import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.inference import model_instance
from app.main import app
from app.schemas import FindingFeatures


SERVICE_DIR = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((SERVICE_DIR / "model_schema.json").read_text(encoding="utf-8"))
CATEGORICALS = set(SCHEMA["categorical_features"])
FLOAT_FEATURES = {
    "finding_start_line_ratio",
}


def make_payload(**overrides):
    payload = {}
    for feature in SCHEMA["api_expected_features"]:
        if feature == "message":
            payload[feature] = "Example lint finding message."
        elif feature in CATEGORICALS:
            payload[feature] = "unseen-test-category"
        elif feature in FLOAT_FEATURES:
            payload[feature] = 0.5
        else:
            payload[feature] = 1
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_predict_single(client):
    response = client.post("/predict", json=make_payload())
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) == {
        "probabilities",
        "ensemble_surface_probability",
        "threshold",
        "decision",
    }
    assert 0.0 <= data["ensemble_surface_probability"] <= 1.0
    assert data["decision"] in {"surface", "suppress"}
    assert data["threshold"] == pytest.approx(0.505)
    assert set(data["probabilities"]) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }


def test_predict_batch(client):
    response = client.post(
        "/predict/batch",
        json={"items": [make_payload(), make_payload(message="second finding")]},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][1]["decision"] in {"surface", "suppress"}


def test_predict_missing_fields(client):
    response = client.post("/predict", json={"rule_id": "no-undef"})
    assert response.status_code == 422


def test_forbidden_extra_field_is_rejected(client):
    response = client.post(
        "/predict",
        json=make_payload(surface_target=1),
    )
    assert response.status_code == 422


def test_model_threshold_and_components_match_artifact(client):
    artifact = joblib.load(SERVICE_DIR / "model" / "eslint_surface_hybrid_ensemble.joblib")
    info = client.get("/model-info")
    assert info.status_code == 200
    data = info.json()
    assert data["threshold"] == pytest.approx(artifact["threshold"])
    assert data["feature_count"] == 55
    assert data["input_feature_count"] == 43
    assert data["model_family"] == "hybrid_ensemble"
    assert data["includes_message_tfidf"] is True
    assert set(data["component_models"]) == set(artifact["models"])


def test_schema_matches_artifact_feature_order(client):
    assert model_instance.features == SCHEMA["expected_features"]
    assert len(model_instance.features) == SCHEMA["feature_count"] == 55
    assert model_instance.api_features == SCHEMA["api_expected_features"]
    assert len(model_instance.api_features) == SCHEMA["api_feature_count"] == 43
    assert set(model_instance.api_features) == set(model_instance.features) - set(
        SCHEMA["features_imputed_by_api"]
    )


def test_43_field_input_uses_fitted_medians_for_omitted_features(client):
    payload = make_payload()
    assert set(payload) == set(SCHEMA["api_expected_features"])
    finding = FindingFeatures.model_validate(payload)
    frame = model_instance._features_to_dataframe([finding])
    assert frame[SCHEMA["features_imputed_by_api"]].isna().all().all()
    transformed = model_instance.preprocessor.transform(frame)
    assert transformed.shape[0] == 1
    assert not set(SCHEMA["metadata_not_used_by_estimator"]) & set(
        model_instance.features
    )


def test_direct_artifact_and_api_scores_match(client):
    artifact = joblib.load(SERVICE_DIR / "model" / "eslint_surface_hybrid_ensemble.joblib")
    payload = make_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text

    finding = model_instance._features_to_dataframe(
        [FindingFeatures.model_validate(payload)]
    )
    transformed = artifact["preprocessor"].transform(finding)
    positive_class = int(artifact["label_encoder"].transform(["surface"])[0])
    component_probabilities = {}
    for name, model in artifact["models"].items():
        class_index = int(np.flatnonzero(np.asarray(model.classes_) == positive_class)[0])
        component_probabilities[name] = model.predict_proba(transformed)[0, class_index]
    weights = artifact["weights"]
    direct_scores = {
        name: float(probability)
        for name, probability in component_probabilities.items()
    }
    direct_score = sum(
        weights[name] * probability
        for name, probability in component_probabilities.items()
    ) / sum(weights.values())
    response_data = response.json()
    assert response_data["ensemble_surface_probability"] == pytest.approx(
        direct_score, abs=1e-6
    )
    assert response_data["probabilities"] == pytest.approx(direct_scores, abs=1e-6)
