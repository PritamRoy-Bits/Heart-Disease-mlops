"""Download the UCI Heart Disease (Cleveland) dataset.

Usage:
    python -m src.data.download_data

Falls back to a clear error message with manual-download instructions if the
UCI server is unreachable (e.g. no internet access in a CI/sandbox).
"""
import logging
import sys

import pandas as pd
import requests

from src.config import COLUMN_NAMES, RAW_FILE, UCI_DATA_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(url: str = UCI_DATA_URL, out_path=RAW_FILE) -> pd.DataFrame:
    """Download the raw CSV (no header) from UCI and save it with proper column names."""
    try:
        logger.info("Downloading dataset from %s", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Could not download dataset automatically: %s", exc)
        logger.error(
            "Manual fallback: download 'processed.cleveland.data' from "
            "https://archive.ics.uci.edu/dataset/45/heart+disease and place it at %s "
            "then re-run this script with --skip-download, or add the 14 columns: %s",
            out_path, COLUMN_NAMES,
        )
        raise

    from io import StringIO

    df = pd.read_csv(StringIO(response.text), header=None, names=COLUMN_NAMES, na_values="?")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved raw dataset with shape %s to %s", df.shape, out_path)
    return df


if __name__ == "__main__":
    try:
        download_dataset()
    except Exception:
        sys.exit(1)
