"""Measure latency of the deployed surface hybrid ensemble API."""

import json
import os
import statistics
import time
from pathlib import Path

import requests


BASE_URL = os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BASE_URL}/predict"
SERVICE_DIR = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (SERVICE_DIR / "model_schema.json").read_text(encoding="utf-8")
)


def sample_payload() -> dict[str, str | int | float]:
    strings = set(SCHEMA["feature_types"]["strings"])
    floats = set(SCHEMA["feature_types"]["floats"])
    return {
        feature: (
            "benchmark-example"
            if feature == "message"
            else "unknown-benchmark-category"
            if feature in strings
            else 0.5
            if feature in floats
            else 1
        )
        for feature in SCHEMA["api_expected_features"]
    }


def main() -> None:
    payload = sample_payload()
    request_count = 100
    warmup = requests.post(API_URL, json=payload, timeout=10)
    warmup.raise_for_status()

    latencies = []
    for _ in range(request_count):
        started = time.perf_counter()
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        latencies.append((time.perf_counter() - started) * 1000)

    latencies.sort()
    p95_index = round(0.95 * (len(latencies) - 1))
    print(f"Threshold: {warmup.json()['threshold']}")
    print(f"Requests: {request_count}")
    print(f"Average latency: {statistics.mean(latencies):.2f} ms")
    print(f"p50 latency: {statistics.median(latencies):.2f} ms")
    print(f"p95 latency: {latencies[p95_index]:.2f} ms")
    print(f"Maximum latency: {max(latencies):.2f} ms")


if __name__ == "__main__":
    main()
