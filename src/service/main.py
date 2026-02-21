from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data import preprocess_pil_image
from src.model import load_checkpoint, predict_probabilities


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference-service")

app = FastAPI(title="Cats vs Dogs Classifier")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "artifacts/labels.json"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUEST_COUNT = 0
TOTAL_LATENCY_SEC = 0.0


@app.on_event("startup")
def startup_event() -> None:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError(
            "Model artifacts missing. Train first or mount artifacts/model.pt and artifacts/labels.json"
        )

    model, labels = load_checkpoint(MODEL_PATH, LABELS_PATH, DEVICE)
    app.state.model = model
    app.state.labels = labels
    logger.info("Model loaded from %s with labels %s", MODEL_PATH, labels)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    global REQUEST_COUNT, TOTAL_LATENCY_SEC

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    REQUEST_COUNT += 1
    TOTAL_LATENCY_SEC += elapsed
    avg_latency = TOTAL_LATENCY_SEC / REQUEST_COUNT
    logger.info(
        "request method=%s path=%s status=%s latency_ms=%.2f total_requests=%s avg_latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed * 1000,
        REQUEST_COUNT,
        avg_latency * 1000,
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True, "requests": REQUEST_COUNT}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        from io import BytesIO
        from PIL import Image

        image = Image.open(BytesIO(contents))
        tensor = preprocess_pil_image(image).to(DEVICE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc

    probabilities = predict_probabilities(app.state.model, tensor, app.state.labels)
    predicted_label = max(probabilities, key=probabilities.get)

    logger.info(
        "prediction filename=%s predicted_label=%s confidence=%.4f",
        file.filename,
        predicted_label,
        probabilities[predicted_label],
    )

    return {
        "predicted_label": predicted_label,
        "probabilities": probabilities,
    }
