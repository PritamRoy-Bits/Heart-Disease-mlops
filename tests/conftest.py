"""Shared pytest fixtures: a small synthetic dataset that mirrors the real schema."""
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_TARGET_COLUMN


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny raw-shaped dataframe, including a missing value, matching the UCI schema."""
    data = {
        "age": [63, 67, 41, 56, 57],
        "sex": [1, 1, 0, 1, 0],
        "cp": [3, 4, 2, 2, 4],
        "trestbps": [145, 160, 130, 120, 140],
        "chol": [233, 286, 204, 236, 241],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [0, 2, 0, 0, 0],
        "thalach": [150, 108, 172, 178, 123],
        "exang": [0, 1, 0, 0, 1],
        "oldpeak": [2.3, 1.5, 1.4, 0.8, 0.2],
        "slope": [0, 1, 2, 2, 1],
        "ca": [0.0, 3.0, None, 0.0, 0.0],
        "thal": [6.0, 3.0, 3.0, 3.0, 7.0],
        RAW_TARGET_COLUMN: [0, 2, 0, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def feature_columns():
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
