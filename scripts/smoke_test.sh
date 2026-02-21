#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-tests/assets/sample.png}"

health_response=$(curl -fsS "$BASE_URL/health")
echo "Health: $health_response"

predict_response=$(curl -fsS -X POST "$BASE_URL/predict" -F "file=@${SAMPLE_IMAGE}")
echo "Predict: $predict_response"

if ! echo "$predict_response" | grep -q "predicted_label"; then
  echo "Smoke test failed: prediction payload missing predicted_label"
  exit 1
fi

echo "Smoke test passed"
