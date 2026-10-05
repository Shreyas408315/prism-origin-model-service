"""Compare live API scores with direct inference from the bundled artifact."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

BASE_URL = os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BASE_URL}/predict"
SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from app.inference import PRismSurfaceClassifier
from app.schemas import FindingFeatures


SCHEMA = json.loads(
    (SERVICE_DIR / "model_schema.json").read_text(encoding="utf-8")
)


def sample_payload() -> dict[str, str | int | float]:
    strings = set(SCHEMA["feature_types"]["strings"])
    floats = set(SCHEMA["feature_types"]["floats"])
    return {
        feature: (
            "verification-example"
            if feature == "message"
            else "unknown-verification-category"
            if feature in strings
            else 0.5
            if feature in floats
            else 1
        for feature in SCHEMA["api_expected_features"]
        )
    }


def main() -> None:
    classifier = PRismSurfaceClassifier()
    classifier.load_model()
    payload = sample_payload()
    finding = FindingFeatures.model_validate(payload)
    frame = classifier._features_to_dataframe([finding])
    transformed = classifier.preprocessor.transform(frame)
    positive_class = int(classifier.label_encoder.transform(["surface"])[0])

    component_scores = {}
    for name, model in classifier.models.items():
        positive_index = int(
            np.flatnonzero(np.asarray(model.classes_) == positive_class)[0]
        )
        component_scores[name] = float(
            model.predict_proba(transformed)[0, positive_index]
        )
    expected_score = sum(
        classifier.weights[name] * score
        for name, score in component_scores.items()
    ) / sum(classifier.weights.values())

    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    actual_score = response.json()["ensemble_surface_probability"]
    difference = abs(expected_score - actual_score)

    print(f"Direct artifact score: {expected_score:.8f}")
    print(f"API score:             {actual_score:.8f}")
    print(f"Absolute difference:   {difference:.8g}")
    if difference > 1e-6:
        raise SystemExit("FAIL: API score differs from direct artifact inference.")
    print("PASS: API matches direct artifact inference.")


if __name__ == "__main__":
    main()
