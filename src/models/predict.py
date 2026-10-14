"""Load the persisted model pipeline and run inference on new patient records."""
import logging

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "heart_disease_model.pkl"

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading model from %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(features: dict) -> dict:
    """Predict heart-disease risk for a single patient record.

    `features` must contain exactly the keys in FEATURE_COLUMNS.
    """
    model = get_model()
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    proba = model.predict_proba(row)[0, 1]
    prediction = int(proba >= 0.5)
    return {
        "prediction": prediction,
        "label": "disease" if prediction == 1 else "no_disease",
        "confidence": float(proba if prediction == 1 else 1 - proba),
    }
