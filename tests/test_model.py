import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN
from src.data.preprocess import clean_data
from src.features.build_features import build_preprocessor
from src.models.train import evaluate


def test_pipeline_trains_and_predicts(raw_df, feature_columns):
    cleaned = clean_data(raw_df)
    X, y = cleaned[feature_columns], cleaned[TARGET_COLUMN]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


def test_evaluate_returns_expected_metric_keys(raw_df, feature_columns):
    cleaned = clean_data(raw_df)
    X, y = cleaned[feature_columns], cleaned[TARGET_COLUMN]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)

    metrics = evaluate(pipeline, X, y)
    assert set(metrics) == {"test_accuracy", "test_precision", "test_recall", "test_roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
