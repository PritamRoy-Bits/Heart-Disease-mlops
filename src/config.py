"""Central configuration for paths, column names, and constants used across the pipeline."""
import os
from pathlib import Path

# Newer MLflow versions (3.x) deprecated the local filesystem tracking backend in favor of a
# database backend. We intentionally keep the simple file-based "mlruns/" store for this
# assignment (no extra DB server to set up), so explicitly opt back in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODELS_DIR = ROOT_DIR / "saved_models"

RAW_FILE = DATA_RAW_DIR / "heart_disease_raw.csv"
PROCESSED_FILE = DATA_PROCESSED_DIR / "heart_disease_processed.csv"

# UCI Heart Disease (Cleveland) dataset - 14 attributes.
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Original target "num" is 0 (no disease) .. 4 (severity levels).
# We binarize it: 0 = no disease, 1 = disease present (num > 0).
TARGET_COLUMN = "target"
RAW_TARGET_COLUMN = "num"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2

MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = f"file:///{(ROOT_DIR / 'mlruns').as_posix()}"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
