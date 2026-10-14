# Heart Disease Risk Prediction — MLOps Project Report

*(Fill in every bracketed section, then export this file to PDF/DOCX for submission —
target length ~10 pages. Delete this instruction line before export.)*

## 1. Introduction & Problem Statement
- Objective, dataset summary (UCI Heart Disease / Cleveland, 303 records, 13 features + target).
- Business framing: why early risk screening matters.

## 2. Setup / Install Instructions
- [Paste the "Environment Setup" and "Running the Pipeline" sections from the
  separate STEPS guide, or link to the repo README.]

## 3. Data & EDA
- Missing-value handling (`ca`, `thal` columns, median imputation).
- Target binarization rationale (`num > 0` -> `target = 1`).
- Insert: `reports/figures/feature_histograms.png`
- Insert: `reports/figures/correlation_heatmap.png`
- Insert: `reports/figures/class_balance.png`
- [Your written observations: which features correlate with disease risk, class imbalance ratio, etc.]

## 4. Feature Engineering & Modelling Choices
- Preprocessing pipeline: `StandardScaler` on numeric features, `OneHotEncoder` on categoricals.
- Models trained: Logistic Regression, Random Forest — rationale for choosing these two.
- Hyperparameter tuning: `GridSearchCV` with 5-fold stratified cross-validation, scoring = ROC-AUC.
- [Your table: best hyperparameters per model.]
- [Your table: cv_best_roc_auc, test_accuracy, test_precision, test_recall, test_roc_auc per model.]
- Insert: `reports/figures/confusion_matrix_<best_model>.png`
- Insert: `reports/figures/roc_curve_<best_model>.png`
- Final model selection justification.

## 5. Experiment Tracking (MLflow)
- Summary of tracked runs (screenshot of `mlflow ui` experiment table).
- What was logged: hyperparameters, CV + test metrics, confusion-matrix/ROC plots, model artifact.
- [Your screenshot here.]

## 6. Model Packaging & Reproducibility
- Model persisted as `saved_models/heart_disease_model.pkl` via `joblib`
  (full sklearn `Pipeline` including the preprocessing step — guarantees the
  same transform is applied at inference time as at training time).
- `requirements.txt` pins exact dependency versions for a reproducible environment.

## 7. CI/CD Pipeline
- GitHub Actions workflow (`.github/workflows/ci.yml`): lint (flake8) -> unit tests (pytest) ->
  data download/preprocess -> EDA -> model training -> Docker build + smoke test.
- [Your screenshot of a green Actions run.]
- Explain how the pipeline fails loudly on lint/test errors (non-zero exit codes, step logs).

## 8. Model Containerization
- Dockerfile: `python:3.11-slim`, installs `requirements.txt`, copies `src/` and `saved_models/`,
  serves via `uvicorn` on port 8000.
- [Your screenshot: `docker build` succeeding.]
- [Your screenshot: `docker run` + a sample `curl -X POST /predict` call and its JSON response.]

## 9. Monitoring & Logging
- Every API request is logged (method, path, status, latency) to `logs/api_requests.log` and stdout.
- `/metrics` endpoint exposes Prometheus-format counters (`api_requests_total`) and
  latency histograms (`api_request_latency_seconds`) for request monitoring.
- [Optional: screenshot of a local Prometheus scrape or Grafana panel, if set up.]

## 10. Architecture Diagram
- [Insert a diagram: Data source -> preprocess -> train+MLflow -> saved_models/ ->
  FastAPI container -> client request/response, with logging/metrics branch.]

## 11. Production Deployment
- *Out of scope for this submission* — the API is verified locally via Docker only
  (see Section 8). No cloud/Kubernetes deployment was performed.

## 12. Conclusion & Future Work
- Summary of results, limitations, and what you'd add next (e.g. drift monitoring,
  cloud deployment, more models).

## Appendix
- Link to GitHub repository: `[your repo URL]`
- [Any additional screenshots or logs.]
