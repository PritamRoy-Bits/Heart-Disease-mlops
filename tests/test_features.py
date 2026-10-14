import pandas as pd

from src.data.preprocess import clean_data
from src.features.build_features import build_preprocessor


def test_preprocessor_outputs_numeric_matrix_only(raw_df, feature_columns):
    cleaned = clean_data(raw_df)
    X = cleaned[feature_columns]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    # Output must be fully numeric (no object dtype leaking through) for sklearn models.
    assert not pd.DataFrame(transformed).select_dtypes(include="object").shape[1]


def test_preprocessor_is_reusable_on_unseen_rows(raw_df, feature_columns):
    cleaned = clean_data(raw_df)
    X = cleaned[feature_columns]
    preprocessor = build_preprocessor()
    preprocessor.fit(X.iloc[:3])
    transformed = preprocessor.transform(X.iloc[3:])
    assert transformed.shape[0] == 2
