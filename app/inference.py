import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from app.schemas import (
    FindingFeatures,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

class PRismOriginClassifier:
    """Production wrapper for the PRism Ensemble Origin Classifier."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Default to model artifact in the model directory
            base_dir = Path(__file__).resolve().parent.parent
            model_path = os.getenv(
                "MODEL_PATH", str(base_dir / "model" / "prism_ensemble_clean.joblib")
            )

        self.model_path = model_path
        self.artifact: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.default_threshold: float = 0.574674670640332
        self.features: List[str] = []
        self.positive_class: str = "INTRODUCED"
        self.is_loaded: bool = False
        self.model_version: str = "prism-origin-ensemble-clean-v1"

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        loaded = joblib.load(self.model_path)
        if not isinstance(loaded, dict) or "models" not in loaded:
            raise ValueError(f"Invalid model artifact format at {self.model_path}")

        self.artifact = loaded
        self.models = loaded["models"]
        self.weights = loaded.get("weights", {k: 1.0 for k in self.models})
        self.default_threshold = float(loaded.get("threshold", 0.574674670640332))
        self.features = loaded.get("features", [])
        self.positive_class = loaded.get("positive_class", "INTRODUCED")
        self.is_loaded = True

    def _features_to_dataframe(self, findings: List[FindingFeatures]) -> pd.DataFrame:
        rows = [f.model_dump() for f in findings]
        df = pd.DataFrame(rows)
        # Ensure only the expected model features in correct order
        return df[self.features]

    def predict_batch(
        self,
        findings: List[FindingFeatures]
    ) -> BatchPredictionResponse:
        if not findings:
            return BatchPredictionResponse(
                model_version=self.model_version,
                threshold=self.default_threshold,
                predictions=[]
            )

        X = self._features_to_dataframe(findings)
        threshold = self.default_threshold

        model_probs = {}
        weighted_sum = np.zeros(len(X))
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            prob = model.predict_proba(X)[:, 1]
            model_probs[name] = prob
            weight = self.weights.get(name, 1.0)
            weighted_sum += weight * prob

        ensemble_proba = weighted_sum / total_weight

        results = []

        for i, finding in enumerate(findings):
            prob = float(ensemble_proba[i])
            is_intro = prob >= threshold

            per_model_result = {
                name: float(model_probs[name][i]) for name in self.models
            }

            results.append(
                PredictionResult(
                    model_version=self.model_version,
                    positive_class=self.positive_class,
                    risk_score=prob,
                    decision="INTRODUCED" if is_intro else "PRE_EXISTING",
                    threshold=threshold,
                    component_scores=per_model_result
                )
            )

        return BatchPredictionResponse(
            model_version=self.model_version,
            threshold=threshold,
            predictions=results
        )

    def predict(self, finding: FindingFeatures) -> PredictionResult:
        batch_resp = self.predict_batch([finding])
        return batch_resp.predictions[0]

# Global singleton instance
model_instance = PRismOriginClassifier()
