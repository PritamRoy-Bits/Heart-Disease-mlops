"""Train and tune Logistic Regression and Random Forest classifiers for heart-disease
prediction, track every run with MLflow, and persist the best pipeline to disk.

Usage:
    python -m src.models.train
"""
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    RocCurveDisplay, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import (  # noqa: E402
    FEATURE_COLUMNS, FIGURES_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_DIR,
    PROCESSED_FILE, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE,
)
from src.features.build_features import build_preprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_GRID = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["liblinear", "lbfgs"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    },
}


def load_split(processed_path=PROCESSED_FILE):
    df = pd.read_csv(processed_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_diagnostic_plots(model, X_test, y_test, run_name: str) -> list:
    """Save confusion-matrix and ROC-curve plots for this run and return their paths."""
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, model.predict(X_test))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {run_name}")
    cm_path = FIGURES_DIR / f"confusion_matrix_{run_name}.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    paths.append(cm_path)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve - {run_name}")
    roc_path = FIGURES_DIR / f"roc_curve_{run_name}.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    paths.append(roc_path)

    return paths


def train_and_log(name: str, spec: dict, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("clf", spec["estimator"])])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(pipeline, spec["param_grid"], cv=cv, scoring="roc_auc", n_jobs=-1)

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)

        test_metrics = evaluate(best_model, X_test, y_test)
        mlflow.log_metrics(test_metrics)

        for plot_path in log_diagnostic_plots(best_model, X_test, y_test, name):
            mlflow.log_artifact(str(plot_path))

        mlflow.sklearn.log_model(best_model, artifact_path="model", serialization_format="pickle")

        logger.info("[%s] best_params=%s cv_roc_auc=%.4f test_metrics=%s",
                    name, search.best_params_, search.best_score_, test_metrics)

        return best_model, {**test_metrics, "cv_best_roc_auc": search.best_score_}


def run() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_split()

    results = {}
    models = {}
    for name, spec in MODEL_GRID.items():
        model, metrics = train_and_log(name, spec, X_train, X_test, y_train, y_test)
        results[name] = metrics
        models[name] = model

    best_name = max(results, key=lambda n: results[n]["test_roc_auc"])
    best_model = models[best_name]
    logger.info("Best model: %s with metrics %s", best_name, results[best_name])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "heart_disease_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Saved best model pipeline to %s", model_path)

    with open(MODELS_DIR / "best_model.txt", "w") as f:
        f.write(f"{best_name}\n{results[best_name]}\n")

    return results


if __name__ == "__main__":
    run()
