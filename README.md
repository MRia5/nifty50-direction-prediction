# NIFTY 50 Direction Prediction

Financial ML project for predicting the next trading day's NIFTY 50 close direction:

`target = 1 if close[t + 1] > close[t], else 0`

The main pipeline is an MLflow-tracked expanding-window workflow using a single LightGBM gradient-boosting classifier.

## Project Structure

```text
data/raw/                         Input CSV files
docs/                             Original brief and data bundle notes
src/financial_ml_project/         Reusable Python package
outputs/                          Generated reports, plots, predictions, metrics
mlruns/                           Committed MLflow experiment runs and model artifacts
main.py                           End-to-end runner
requirements.txt                  Python dependencies
tests/                            Lightweight unit tests
```

## How To Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Modeling Approach

- Model: `lightgbm.LGBMClassifier`
- Removed: logistic regression, random forest, and sklearn `GradientBoostingClassifier`
- Features: exactly 12 audited same-day/trailing features from `starter_features.csv`
- In-sample walk-forward: January 2022 through June 2025
- Out-of-sample lockbox: July 2025 through December 2025
- Validation: expanding monthly folds, train months 1-N and predict month N+1
- Tracking: every training fold is wrapped in `mlflow.start_run()` before fitting

## Outputs

Running `python main.py` writes:

- `outputs/report.md`
- `outputs/feature_audit.csv`
- `outputs/walk_forward_folds.csv`
- `outputs/walk_forward_predictions.csv`
- `outputs/oos_predictions.csv`
- `outputs/metrics_with_bootstrap_ci.csv`
- `outputs/feature_importance.csv`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `mlruns/`

## Metrics

The model is compared against a majority-class baseline using:

- AUC
- Balanced accuracy
- Confusion matrix counts
- Sharpe ratio
- Max drawdown
- Hit rate
- Turnover

Each reported metric includes a 95 percent bootstrap confidence interval.

## Feature Audit

The project audits every column in `starter_features.csv`. Direct future-looking column names were not found, but opaque signal-like, sparse, and overlapping features are dropped. The final feature list and one-line justification for each feature are written into `outputs/report.md` and `outputs/feature_audit.csv`.
