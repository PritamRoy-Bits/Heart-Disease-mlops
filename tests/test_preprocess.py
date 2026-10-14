import pandas as pd

from src.config import TARGET_COLUMN
from src.data.preprocess import clean_data


def test_clean_data_fills_missing_values(raw_df):
    assert raw_df["ca"].isna().any()
    cleaned = clean_data(raw_df)
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_binarizes_target(raw_df):
    cleaned = clean_data(raw_df)
    assert TARGET_COLUMN in cleaned.columns
    assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1})
    # original num=2 and num=1 rows must map to target=1
    assert cleaned[TARGET_COLUMN].sum() == 2


def test_clean_data_drops_raw_target_column(raw_df):
    cleaned = clean_data(raw_df)
    assert "num" not in cleaned.columns


def test_clean_data_removes_duplicates(raw_df):
    duped = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(duped)
    assert cleaned.duplicated().sum() == 0
