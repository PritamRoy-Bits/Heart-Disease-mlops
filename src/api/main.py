"""FastAPI model-serving app for heart-disease risk prediction.

Endpoints:
    GET  /health   - liveness probe
    POST /predict  - JSON patient features -> prediction + confidence
    GET  /metrics  - Prometheus-format metrics for monitoring

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import PatientFeatures, PredictionResponse
from src.models.predict import predict_one

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "api_requests.log"), logging.StreamHandler()],
    force=True,  # other already-imported modules (e.g. src.models.predict) may have
                 # called basicConfig() first; without force=True that earlier call
                 # wins and this FileHandler is silently never attached.
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])


@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    """Log every request with method/path/status/latency and expose it via Prometheus."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(endpoint=request.url.path, status=response.status_code).inc()
    logger.info("%s %s -> %s (%.4fs)", request.method, request.url.path,
                response.status_code, duration)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    try:
        result = predict_one(features.model_dump())
    except FileNotFoundError as exc:
        logger.error("Model artifact not found: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
