# NIFTY 50 Direction Prediction

Python machine-learning project for the Financial ML track. The goal is to predict the next trading day's NIFTY 50 close direction:

`target = 1 if close[t + 1] > close[t], else 0`

The project uses the provided 2022-2025 NSE data bundle, starter technical/cross-asset features, and a chronological train/test split so future observations never leak into the model.

## Project Structure

```text
data/raw/                         Input CSV files
docs/                             Original brief and data bundle notes
src/financial_ml_project/         Reusable Python package
outputs/                          Generated metrics, predictions, plots, report
main.py                           End-to-end runner
requirements.txt                  Python dependencies
tests/                            Lightweight unit tests
```

## How To Run

Create a virtual environment, install dependencies, then run the pipeline.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The run writes:

- `outputs/metrics.csv`
- `outputs/predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/report.md`

## Modeling Approach

The pipeline compares three baseline classifiers:

- Logistic regression with scaling
- Random forest
- Gradient boosting

Rows are sorted by date, features are imputed inside each model pipeline, and the last 20 percent of observations are held out as the final test set. Model selection is based on walk-forward cross-validation accuracy, with final evaluation reported on the untouched chronological test period.

## Feature Set

The implementation uses `starter_features.csv` as the main feature table because it already combines NIFTY price behavior, Bank Nifty cross-asset signals, India VIX information, rolling volatility, volume, and calendar effects. The code still rebuilds the target directly from `nifty50.csv` so the target definition is explicit and auditable.

Features with more than 25 percent missing values are dropped before modeling. This removes long-lookback columns that are unavailable in the early walk-forward folds.

## Notes

This is a directional classification project, not an executable trading strategy. Accuracy, ROC-AUC, precision, recall, and confusion matrix results should be interpreted as model diagnostics rather than evidence of profitability after costs, slippage, and position sizing.
