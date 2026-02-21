#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-tests/assets/sample.png}"

max_attempts=30
attempt=1

until health_response=$(curl -fsS "$BASE_URL/health"); do
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "Smoke test failed: health endpoint not reachable after ${max_attempts} attempts"
    exit 1
  fi
  echo "Waiting for service readiness (attempt ${attempt}/${max_attempts})..."
  attempt=$((attempt + 1))
  sleep 2
done

echo "Health: $health_response"

attempt=1
until predict_response=$(curl -fsS -X POST "$BASE_URL/predict" -F "file=@${SAMPLE_IMAGE}"); do
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "Smoke test failed: predict endpoint failed after ${max_attempts} attempts"
    exit 1
  fi
  echo "Prediction endpoint not ready yet (attempt ${attempt}/${max_attempts})..."
  attempt=$((attempt + 1))
  sleep 2
done

echo "Predict: $predict_response"

if ! echo "$predict_response" | grep -q "predicted_label"; then
  echo "Smoke test failed: prediction payload missing predicted_label"
  exit 1
fi

echo "Smoke test passed"
