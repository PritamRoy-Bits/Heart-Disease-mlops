"""Clean the raw UCI Heart Disease data: handle missing values and binarize the target."""
import logging

import pandas as pd

from src.config import PROCESSED_FILE, RAW_FILE, RAW_TARGET_COLUMN, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric/categorical values and binarize the multi-class target."""
    df = df.copy()

    # "ca" and "thal" are the only columns with missing values ("?" -> NaN) in this dataset.
    for col in df.columns:
        if df[col].isna().any():
            median = df[col].median()
            logger.info("Imputing %d missing value(s) in '%s' with median=%s",
                        df[col].isna().sum(), col, median)
            df[col] = df[col].fillna(median)

    df[TARGET_COLUMN] = (df[RAW_TARGET_COLUMN] > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET_COLUMN])

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def run(raw_path=RAW_FILE, out_path=PROCESSED_FILE) -> pd.DataFrame:
    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    cleaned = clean_data(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    logger.info("Saved cleaned dataset with shape %s to %s", cleaned.shape, out_path)
    return cleaned


if __name__ == "__main__":
    run()
