# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end ML pipeline for the UCI Heart Disease (Cleveland) dataset: data
acquisition, EDA, feature engineering, model training with MLflow tracking,
automated testing/CI, and a Dockerized FastAPI serving layer.

> Cloud/Kubernetes deployment is intentionally out of scope for this stage.

## Project layout

```
heart-disease-mlops/
├── data/
│   ├── raw/                  # raw dataset, as downloaded from UCI
│   └── processed/            # cleaned dataset used for training
├── src/
│   ├── config.py             # paths, column names, constants
│   ├── data/                 # download, preprocess, EDA scripts
│   ├── features/             # reusable preprocessing pipeline
│   ├── models/               # train, evaluate, predict
│   └── api/                  # FastAPI serving app
├── notebooks/                # exploratory notebook
├── tests/                    # pytest unit tests
├── reports/figures/          # EDA + evaluation plots
├── saved_models/             # persisted model pipeline (joblib)
├── .github/workflows/ci.yml  # lint -> test -> train -> docker build CI
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.data.download_data
python -m src.data.preprocess
python -m src.data.eda
python -m src.models.train
pytest tests/ -v
uvicorn src.api.main:app --reload
```

See the separate setup/reproduction guide for full step-by-step instructions.

## API

`POST /predict` — accepts the 13 UCI Heart Disease features as JSON, returns
`{"prediction": 0|1, "label": "...", "confidence": 0.0-1.0}`.

`GET /metrics` — Prometheus-format request count/latency metrics.

## Experiment tracking

All training runs (params, metrics, plots, model artifact) are logged with
MLflow to the local `mlruns/` store. Inspect them with:

```bash
mlflow ui --backend-store-uri mlruns
```
