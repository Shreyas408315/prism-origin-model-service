import os
import joblib
import pandas as pd
import requests
import json
import time
from pathlib import Path

BASE_URL = os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BASE_URL}/predict"

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = SCRIPT_DIR.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(SERVICE_DIR / "model" / "prism_ensemble_clean.joblib"))

# Check possible locations for CSV
csv_candidates = [
    SERVICE_DIR.parent / "ml" / "eslint_ml_features_introduced_v1.csv",
    SERVICE_DIR / "eslint_ml_features_introduced_v1.csv",
    Path("model/eslint_ml_features_introduced_v1.csv"),
]
CSV_PATH = next((str(p) for p in csv_candidates if p.exists()), str(SERVICE_DIR.parent / "ml" / "eslint_ml_features_introduced_v1.csv"))

def main():
    print(f"Loading artifact directly from: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    models = artifact["models"]
    weights = artifact["weights"]
    features = artifact["features"]
    
    print(f"Loading test data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).head(1)
    
    X = df[features].copy()
    
    # manual bool cast for direct inference
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    # direct inference
    probs = [weights[n] * m.predict_proba(X)[:, 1] for n, m in models.items()]
    total_w = sum(weights.values())
    direct_score = float(sum(probs)[0] / total_w)
    
    # API inference
    print("Calling API...")
    payload = X.iloc[0].to_dict()
    # convert any numpy ints to python ints
    payload = {k: int(v) if isinstance(v, (pd.Series, pd.Index, int, float)) and v == int(v) else v for k, v in payload.items()}
    # stringify rule_id and family just in case
    payload["rule_id"] = str(payload["rule_id"])
    payload["rule_family"] = str(payload["rule_family"])

    start_t = time.time()
    resp = requests.post(API_URL, json=payload)
    api_time = time.time() - start_t
    
    if resp.status_code != 200:
        print(f"API Error: {resp.status_code} - {resp.text}")
        return
    
    api_score = resp.json()["risk_score"]
    
    diff = abs(direct_score - api_score)
    print(f"Direct score: {direct_score}")
    print(f"API score:    {api_score}")
    print(f"Difference:   {diff}")
    
    if diff < 1e-4:
        print("[PASS] SUCCESS: API output matches direct artifact inference!")
    else:
        print("[FAIL] API output diverges from direct inference.")
        exit(1)

if __name__ == "__main__":
    main()
