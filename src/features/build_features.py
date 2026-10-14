"""Build the reusable preprocessing pipeline (scaling + one-hot encoding).

The returned ColumnTransformer is fit as part of each model's sklearn Pipeline,
so the exact same transformation is applied at train and inference time -
this is what guarantees reproducibility end to end.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numeric features and one-hot encodes categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
        ]
    )
