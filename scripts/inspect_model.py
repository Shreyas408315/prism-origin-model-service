"""Inspect the configured ESLint surface hybrid ensemble artifact."""

import json
import os
from pathlib import Path

import joblib


def main() -> None:
    service_dir = Path(__file__).resolve().parents[1]
    model_path = Path(
        os.environ.get(
            "MODEL_PATH",
            service_dir / "model" / "eslint_surface_hybrid_ensemble.joblib",
        )
    )
    schema = json.loads((service_dir / "model_schema.json").read_text(encoding="utf-8"))
    artifact = joblib.load(model_path)

    print(f"Artifact: {model_path}")
    print(f"Model version: {schema['model_version']}")
    print(f"Task: {schema['task']}")
    print(f"Positive class: {schema['positive_class']}")
    print(f"Threshold: {artifact['threshold']}")
    print(f"Artifact feature count: {len(artifact['categorical_cols']) + len(artifact['numeric_cols']) + 1}")
    print(f"API input feature count: {schema['api_feature_count']}")
    print(f"Text feature: message (TF-IDF)")
    print(f"Categorical features: {artifact['categorical_cols']}")
    print(f"Numeric feature count: {len(artifact['numeric_cols'])}")
    print(f"Components: { {name: type(model).__name__ for name, model in artifact['models'].items()} }")
    print(f"Weights: {artifact['weights']}")
    print(f"Label classes: {list(artifact['label_encoder'].classes_)}")


if __name__ == "__main__":
    main()
