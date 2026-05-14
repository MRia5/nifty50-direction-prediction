from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_metrics(metrics: dict[str, float], model_name: str, cv_results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_frame = pd.DataFrame(
        [{"model": model_name, **metrics}]
    )
    metric_frame.to_csv(output_dir / "metrics.csv", index=False)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)


def save_diagnostic_plots(model, split, confusion, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Down", "Up"]).plot(ax=ax)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, split.x_test, split.y_test, ax=ax)
    ax.set_title("Test ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    model_name: str,
    metrics: dict[str, float],
    cv_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
    n_rows: int,
    train_rows: int,
    test_rows: int,
) -> None:
    top_features = feature_importance.head(10)
    cv_table = cv_results.to_markdown(index=False, floatfmt=".4f")
    feature_table = top_features.to_markdown(index=False, floatfmt=".6f")

    report = f"""# Financial ML Track Report

## Objective

Predict whether the NIFTY 50 close will rise on the next trading day using daily market, volume, volatility, Bank Nifty, and India VIX features.

## Data

- Total usable observations: {n_rows}
- Training observations: {train_rows}
- Test observations: {test_rows}
- Target: `1` when `close[t + 1] > close[t]`, otherwise `0`

The split is chronological. The final 20 percent of observations are used only for out-of-sample testing.

## Models Compared

{cv_table}

Selected model: `{model_name}`.

## Test Performance

| Metric | Value |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1"]:.4f} |
| ROC-AUC | {metrics["roc_auc"]:.4f} |
| Actual up-day rate | {metrics["positive_rate_actual"]:.4f} |
| Predicted up-day rate | {metrics["positive_rate_predicted"]:.4f} |

## Most Important Features

{feature_table}

## Interpretation

This is a supervised directional model rather than a trading system. The most useful next step would be to evaluate turnover, transaction costs, position sizing, and drawdowns before treating the signal as tradable.
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
