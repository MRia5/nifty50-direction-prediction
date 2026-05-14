from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from financial_ml_project.features import FEATURE_JUSTIFICATIONS, SELECTED_FEATURES


def save_confusion_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        predictions["actual"],
        predictions["predicted"],
        labels=[0, 1],
        display_labels=["Down", "Up"],
        ax=ax,
    )
    ax.set_title("Out-of-Sample Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(
        predictions["actual"],
        predictions["probability_up"],
        ax=ax,
    )
    ax.set_title("Out-of-Sample ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def metric_table(metrics: pd.DataFrame, sample: str) -> str:
    subset = metrics[metrics["sample"] == sample].copy()
    subset["value_ci"] = subset.apply(
        lambda row: f"{row['value']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
        axis=1,
    )
    pivot = subset.pivot(index="metric", columns="model", values="value_ci").reset_index()
    return pivot.to_markdown(index=False)


def write_report(
    output_dir: Path,
    fold_summary: pd.DataFrame,
    metrics_with_ci: pd.DataFrame,
    feature_audit: pd.DataFrame,
    feature_importance: pd.DataFrame,
    n_rows: int,
    n_folds: int,
    oos_rows: int,
) -> None:
    selected_rows = [
        {"feature": feature, "justification": FEATURE_JUSTIFICATIONS[feature]}
        for feature in SELECTED_FEATURES
    ]
    selected_table = pd.DataFrame(selected_rows).to_markdown(index=False)
    dropped_table = feature_audit[~feature_audit["selected"]][
        ["feature", "leakage_risk", "reason"]
    ].to_markdown(index=False)
    fold_table = fold_summary[
        ["fold_id", "train_start", "train_end", "test_start", "test_end", "hit_rate", "auc"]
    ].to_markdown(index=False, floatfmt=".4f")
    importance_table = feature_importance.head(12).to_markdown(index=False)

    report = f"""# Financial ML Track Report

## Objective

Predict next-day NIFTY 50 close direction with a single gradient-boosting model and MLflow-tracked expanding-window validation.

## Data And Splits

- Total usable observations: {n_rows}
- Walk-forward in-sample period: January 2022 through June 2025
- Locked out-of-sample period: July 2025 through December 2025
- Expanding walk-forward folds: {n_folds}
- Out-of-sample observations: {oos_rows}

The model trains on months 1-N, predicts month N+1, shifts forward, and repeats. The July-December 2025 block is evaluated once at the end only.

## Model

Only one model is used: `lightgbm.LGBMClassifier`. Logistic regression, random forest, and sklearn's GradientBoostingClassifier were removed. Every walk-forward fold and the final out-of-sample fit are wrapped in `mlflow.start_run()` and logged under the committed `mlruns/` directory.

## Final 12 Features

{selected_table}

## Feature Audit

Every column in `starter_features.csv` was reviewed. No direct tomorrow-close or future-target feature name was found. Opaque prebuilt signal columns and sparse/overlapping columns were dropped.

{dropped_table}

## Walk-Forward Fold Metrics

{fold_table}

## In-Sample Walk-Forward Metrics With 95% Bootstrap CI

{metric_table(metrics_with_ci, "walk_forward")}

## Locked Out-Of-Sample Metrics With 95% Bootstrap CI

{metric_table(metrics_with_ci, "out_of_sample")}

## Feature Importance

{importance_table}

## Interpretation

The majority-class baseline is included beside the model for every reported metric. Trading diagnostics use a simple directional strategy: predicted up is long NIFTY for the next day, predicted down is short NIFTY for the next day. Sharpe, drawdown, and turnover therefore describe signal behavior, not a production trading strategy with costs or constraints.
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
