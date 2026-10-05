# PRism ESLint Surface Classification Service

FastAPI service for the `eslint_surface_hybrid_ensemble.joblib` artifact. It combines Logistic Regression (weight 0.15), Random Forest (0.40), and XGBoost (0.45) to estimate whether a finding should be **SURFACE**d or **SUPPRESS**ed.

## Artifact and contract

- Default model: `model/eslint_surface_hybrid_ensemble.joblib`.
- Startup checks the artifact SHA-256 against `model_schema.json`.
- Model version: `eslint-surface-hybrid-ensemble-v1`.
- Positive class: `surface`; negative class: `suppressed`.
- Frozen artifact threshold: `0.5050000000000001`.
- Inputs: 43 fields, listed as `api_expected_features` in `model_schema.json` (message text, five categoricals, and 37 numeric/context features). The artifact itself was trained with 55 features.
- Preprocessing: the fitted artifact handles TF-IDF text, categorical one-hot encoding, numerical imputation/scaling, and the weighted model components.

The API requires the 43 fields shown in the user's input example. It does not calculate the 12 additional training features (`change_density`, `distance_ratio`, `distance_per_changed_line`, `file_finding_density`, `rule_finding_density`, `finding_to_change_ratio`, `near_change`, `severity_overlap_interaction`, `fix_overlap_interaction`, `suggestion_overlap_interaction`, `change_density_overlap`, and `rule_concentration`). Since their exact formulas are not present, the API leaves these model columns missing and relies on the artifact's fitted median imputer. This is explicit median imputation, not formula-based feature engineering; predictions should be interpreted with that limitation. Target and repository identifiers remain rejected as extra request fields.

All existing labels are synthetic. This is a development candidate and is not approved for production or validated as real-world ESLint surfacing quality.

## Local Windows setup

The artifact contains scikit-learn 1.8.0 serialized estimators, so the dependency is pinned to that version to match the artifact.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The service loads the model once on startup. If loading or artifact validation fails, startup fails loudly.

## Endpoints

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict/batch` (maximum 500 findings per request)
- Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`

Response for `POST /predict`:

```json
{
  "probabilities": {
    "logistic_regression": 0.6384,
    "random_forest": 0.5878,
    "xgboost": 0.5165
  },
  "ensemble_surface_probability": 0.5633,
  "threshold": 0.5050000000000001,
  "decision": "surface"
}
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\
```

Tests compare the API score to direct preprocessing and weighted model inference, check unknown categories and request validation, and do not score held-out labels.

## Docker / Render

The Docker image copies `app/`, the configured hybrid artifact, and `model_schema.json`. Build and run from this directory:

```powershell
docker build -t prism-surface-model-service .
docker run -p 8000:8000 prism-surface-model-service
```

Set `MODEL_PATH` to override the default artifact path. Keep the matching schema and requirements with the service package.
