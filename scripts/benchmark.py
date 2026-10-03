import os
import time
import requests
import statistics
import json
from pathlib import Path
import joblib

BASE_URL = os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BASE_URL}/predict"

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = SCRIPT_DIR.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(SERVICE_DIR / "model" / "prism_ensemble_clean.joblib"))

def measure_model_load_time():
    if os.path.exists(MODEL_PATH):
        t0 = time.time()
        _ = joblib.load(MODEL_PATH)
        return (time.time() - t0) * 1000
    return 0.0

def main():
    print(f"Measuring model load time from {MODEL_PATH}...")
    load_time_ms = measure_model_load_time()

    # Valid feature payload matching the schema
    payload = {
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

    n_requests = 100
    latencies = []

    print(f"Benchmarking {API_URL} with {n_requests} requests...")
    
    # Warmup
    warmup_res = requests.post(API_URL, json=payload)
    if warmup_res.status_code != 200:
        print(f"Error connecting to {API_URL}: {warmup_res.status_code} - {warmup_res.text}")
        return

    for _ in range(n_requests):
        start_t = time.time()
        requests.post(API_URL, json=payload)
        latencies.append(time.time() - start_t)

    latencies_ms = sorted([l * 1000 for l in latencies])
    
    # Calculate percentiles
    avg_latency = statistics.mean(latencies_ms)
    p50_latency = statistics.median(latencies_ms)
    # p95 calculation
    k = 0.95 * (len(latencies_ms) - 1)
    f = int(k)
    c = min(f + 1, len(latencies_ms) - 1)
    p95_latency = latencies_ms[f] + (k - f) * (latencies_ms[c] - latencies_ms[f])
    max_latency = max(latencies_ms)
    min_latency = min(latencies_ms)

    print("\n--- Benchmark Results ---")
    print(f"Model Load Time: {load_time_ms:.2f} ms")
    print(f"Requests: {n_requests}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"p50 Latency: {p50_latency:.2f} ms")
    print(f"p95 Latency: {p95_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")
    print(f"Min Latency: {min_latency:.2f} ms")

if __name__ == "__main__":
    main()

