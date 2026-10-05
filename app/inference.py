"""Inference wrapper for the frozen ESLint surface hybrid ensemble."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import (
    BatchPredictionResponse,
    FindingFeatures,
    PredictionResult,
)


MODEL_VERSION = "eslint-surface-hybrid-ensemble-v1"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "eslint_surface_hybrid_ensemble.joblib"
)
EXPECTED_MODELS = {"logistic_regression", "random_forest", "xgboost"}


class PRismSurfaceClassifier:
    """Apply stored preprocessing and weighted probabilities to surface findings."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(
            model_path or os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self.artifact: dict[str, Any] = {}
        self.models: dict[str, Any] = {}
        self.weights: dict[str, float] = {}
        self.features: list[str] = []
        self.api_features: list[str] = []
        self.preprocessor: Any = None
        self.label_encoder: Any = None
        self.default_threshold = 0.0
        self.model_version = MODEL_VERSION
        self.positive_class = "surface"
        self.is_loaded = False

    def load_model(self) -> None:
        """Load and validate artifact contents; repeated startup calls are harmless."""
        if self.is_loaded:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        service_dir = Path(__file__).resolve().parent.parent
        schema_path = service_dir / "model_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if digest != schema.get("artifact_sha256"):
            raise ValueError("Model artifact checksum does not match model_schema.json.")

        loaded = joblib.load(self.model_path)
        required_keys = {
            "models",
            "weights",
            "threshold",
            "preprocessor",
            "label_encoder",
            "categorical_cols",
            "numeric_cols",
            "drop_cols",
        }
        if not isinstance(loaded, dict) or not required_keys.issubset(loaded):
            missing = sorted(required_keys - set(loaded)) if isinstance(loaded, dict) else []
            raise ValueError(f"Invalid hybrid ensemble artifact; missing keys: {missing}")

        models = loaded["models"]
        if not isinstance(models, dict) or set(models) != EXPECTED_MODELS:
            raise ValueError("Artifact must contain Logistic Regression, Random Forest, and XGBoost.")
        weights = {name: float(value) for name, value in loaded["weights"].items()}
        if set(weights) != EXPECTED_MODELS or any(
            not np.isfinite(weight) or weight < 0 for weight in weights.values()
        ):
            raise ValueError("Artifact ensemble weights must be finite, nonnegative, and complete.")
        if sum(weights.values()) <= 0:
            raise ValueError("Artifact ensemble weights must sum to a positive value.")

        label_encoder = loaded["label_encoder"]
        classes = [str(label) for label in label_encoder.classes_]
        if set(classes) != {"surface", "suppressed"}:
            raise ValueError(f"Unexpected label classes in artifact: {classes}")
        positive_encoded = int(label_encoder.transform(["surface"])[0])
        for name, model in models.items():
            model_classes = np.asarray(model.classes_)
            if positive_encoded not in model_classes:
                raise ValueError(
                    f"Model {name} does not expose the encoded surface class."
                )

        preprocessor = loaded["preprocessor"]
        categorical = list(loaded["categorical_cols"])
        numeric = list(loaded["numeric_cols"])
        if not hasattr(preprocessor, "feature_names_in_"):
            raise ValueError("Artifact preprocessor has no fitted input-column schema.")
        transformer_columns = {
            name: list(columns)
            for name, _, columns in preprocessor.transformers_
            if name in {"text_tfidf", "categorical", "numeric"}
        }
        if transformer_columns.get("text_tfidf") != ["message"]:
            raise ValueError("Artifact text transformer must consume the message column.")
        if transformer_columns.get("categorical") != categorical:
            raise ValueError("Artifact categorical columns do not match its preprocessor.")
        if transformer_columns.get("numeric") != numeric:
            raise ValueError("Artifact numeric columns do not match its preprocessor.")

        self.artifact = loaded
        self.models = models
        self.weights = weights
        self.features = ["message", *categorical, *numeric]
        if self.features != schema.get("expected_features"):
            raise ValueError("Artifact feature order does not match model_schema.json.")
        self.api_features = list(schema.get("api_expected_features", []))
        if len(self.api_features) != schema.get("api_feature_count"):
            raise ValueError("API feature count does not match model_schema.json.")
        if set(self.api_features) != set(self.features) - set(
            schema.get("features_imputed_by_api", [])
        ):
            raise ValueError("API feature set does not match artifact feature schema.")
        if len(self.features) != schema.get("feature_count"):
            raise ValueError("Artifact feature count does not match model_schema.json.")
        if not np.isclose(
            float(loaded["threshold"]),
            float(schema.get("decision_threshold", -1)),
        ):
            raise ValueError("Artifact threshold does not match model_schema.json.")
        if any(
            not np.isclose(self.weights[name], schema["component_weights"][name])
            for name in self.weights
        ):
            raise ValueError("Artifact weights do not match model_schema.json.")

        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.default_threshold = float(loaded["threshold"])
        self.positive_class = "surface"
        self.is_loaded = True

    def _features_to_dataframe(
        self,
        findings: list[FindingFeatures],
    ) -> pd.DataFrame:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        rows = [finding.model_dump() for finding in findings]
        model_frame = pd.DataFrame(rows, columns=self.api_features)
        expected_input_columns = list(self.preprocessor.feature_names_in_)
        frame = pd.DataFrame(
            np.nan,
            index=model_frame.index,
            columns=expected_input_columns,
        )
        for feature in self.api_features:
            frame[feature] = model_frame[feature]
        return frame

    def predict_batch(
        self,
        findings: list[FindingFeatures],
    ) -> BatchPredictionResponse:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        if not findings:
            return BatchPredictionResponse(
                model_version=MODEL_VERSION,
                threshold=self.default_threshold,
                predictions=[],
            )

        model_frame = self._features_to_dataframe(findings)
        transformed = self.preprocessor.transform(model_frame)
        positive_encoded = int(self.label_encoder.transform(["surface"])[0])

        component_scores: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            positive_index = int(
                np.flatnonzero(np.asarray(model.classes_) == positive_encoded)[0]
            )
            component_scores[name] = model.predict_proba(transformed)[:, positive_index]

        total_weight = sum(self.weights.values())
        ensemble_scores = sum(
            self.weights[name] * scores
            for name, scores in component_scores.items()
        ) / total_weight

        predictions = []
        for index, finding in enumerate(findings):
            risk_score = float(ensemble_scores[index])
            predictions.append(
                PredictionResult(
                    probabilities={
                        name: float(scores[index])
                        for name, scores in component_scores.items()
                    },
                    ensemble_surface_probability=risk_score,
                    threshold=self.default_threshold,
                    decision=(
                        "surface"
                        if risk_score >= self.default_threshold
                        else "suppress"
                    ),
                )
            )

        return BatchPredictionResponse(
            model_version=MODEL_VERSION,
            threshold=self.default_threshold,
            predictions=predictions,
        )

    def predict(self, finding: FindingFeatures) -> PredictionResult:
        response = self.predict_batch([finding])
        return response.predictions[0]


# The service lifecycle loads this singleton exactly once at startup.
model_instance = PRismSurfaceClassifier()
