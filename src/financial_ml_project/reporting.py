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


def metric_value(metrics: pd.DataFrame, sample: str, model: str, metric: str) -> float:
    row = metrics[
        (metrics["sample"] == sample)
        & (metrics["model"] == model)
        & (metrics["metric"] == metric)
    ]
    return float(row.iloc[0]["value"])


def diagnostic_value(edge_diagnostics: pd.DataFrame, test: str, statistic: str) -> float:
    row = edge_diagnostics[
        (edge_diagnostics["test"] == test)
        & (edge_diagnostics["statistic"] == statistic)
    ]
    return float(row.iloc[0]["value"])


def write_report(
    output_dir: Path,
    fold_summary: pd.DataFrame,
    metrics_with_ci: pd.DataFrame,
    feature_audit: pd.DataFrame,
    feature_importance: pd.DataFrame,
    edge_diagnostics: pd.DataFrame,
    boundary_sensitivity: pd.DataFrame,
    frame: pd.DataFrame,
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
    diagnostics_table = edge_diagnostics.to_markdown(index=False, floatfmt=".4f")
    sensitivity_table = boundary_sensitivity.to_markdown(index=False, floatfmt=".4f")

    wf_auc = metric_value(metrics_with_ci, "walk_forward", "lightgbm", "auc")
    wf_baseline_auc = metric_value(metrics_with_ci, "walk_forward", "majority_baseline", "auc")
    wf_hit = metric_value(metrics_with_ci, "walk_forward", "lightgbm", "hit_rate")
    wf_baseline_hit = metric_value(metrics_with_ci, "walk_forward", "majority_baseline", "hit_rate")
    oos_hit = metric_value(metrics_with_ci, "out_of_sample", "lightgbm", "hit_rate")
    oos_baseline_hit = metric_value(metrics_with_ci, "out_of_sample", "majority_baseline", "hit_rate")
    oos_auc = metric_value(metrics_with_ci, "out_of_sample", "lightgbm", "auc")
    oos_sharpe = metric_value(metrics_with_ci, "out_of_sample", "lightgbm", "sharpe_ratio")
    oos_baseline_sharpe = metric_value(metrics_with_ci, "out_of_sample", "majority_baseline", "sharpe_ratio")
    edge_p_value = diagnostic_value(edge_diagnostics, "paired_permutation_edge", "p_value")
    random_auc_p_value = diagnostic_value(edge_diagnostics, "random_label_baseline", "auc_p_value")

    positive_rate = frame["target"].mean()
    majority_baseline_accuracy = max(positive_rate, 1 - positive_rate)
    flat_count = int((frame["next_return"] == 0).sum())
    near_zero_count = int((frame["next_return"].abs() < 0.0005).sum())

    report = f"""# Financial ML Track Research Report

## 1. Problem Framing

The task is to predict whether the next NIFTY 50 close is above the current close. I define the label as `1` when `close[t+1] > close[t]` and `0` otherwise. Flat returns are therefore treated as down/non-up days. I did not remove near-zero moves, because removing them would make the target easier after observing the next-day return. In this dataset there are {flat_count} exactly flat next-day returns and {near_zero_count} near-zero returns with absolute next-day return below 5 basis points.

The unconditional up-day rate in the usable sample is {positive_rate:.2%}, so a majority-class classifier has a full-sample baseline accuracy of {majority_baseline_accuracy:.2%}. In the locked July-December 2025 test period, the majority-class baseline accuracy is {oos_baseline_hit:.2%}. This is the minimum hurdle: a useful model should beat this baseline out of sample and should do so by enough that random variation is an implausible explanation.

## 2. Data, Split Design, And Tracking

- Total usable observations: {n_rows}
- In-sample walk-forward period: January 2022 through June 2025
- Locked out-of-sample period: July 2025 through December 2025
- Expanding walk-forward folds: {n_folds}
- Out-of-sample observations: {oos_rows}

The validation loop trains on months 1-N, predicts month N+1, shifts forward, and repeats. The July-December 2025 block is locked away until the final evaluation. Every training fold is wrapped in `mlflow.start_run()` from the beginning of the fold, with parameters, metrics, prediction artifacts, and model artifacts written to the committed `mlruns/` directory.

## 3. Feature Audit And Final Features

I used exactly 12 features. The selection rule was conservative: use features that can plausibly be known at today's close, are trailing/same-day rather than forward-looking, and are interpretable enough to defend. I did not find an explicit tomorrow-close column. I still dropped opaque engineered columns where the construction was not auditable.

{selected_table}

The main suspicious column was `ma5_smooth_signal`. It is not a raw market measurement such as return, volatility, volume, or VIX. It is a prebuilt "signal" column, its exact formula is not documented in the data bundle, and its name suggests a smoothed trading rule rather than a transparent feature. Because the assignment asks for leakage awareness and because the provided starter features were explicitly "not rigorously audited," I treated it as medium leakage/model-design risk and removed it.

Dropped feature audit:

{dropped_table}

## 4. Model

The model is a single `lightgbm.LGBMClassifier`. I removed logistic regression, random forest, and sklearn's `GradientBoostingClassifier` so the experiment is not a model-selection exercise disguised as a final result. Hyperparameters are intentionally modest: shallow trees, small leaves, regularization, and subsampling to reduce overfitting on a short financial time series.

## 5. Results

Walk-forward performance is weak. The model's walk-forward AUC is {wf_auc:.3f}, compared with {wf_baseline_auc:.3f} for the majority baseline. The model's walk-forward hit rate is {wf_hit:.2%}, while the majority baseline hit rate is {wf_baseline_hit:.2%}. In plain English: before the locked test period, the model is not convincingly better than a naive class-prior rule.

Walk-forward fold details:

{fold_table}

In-sample walk-forward metrics with 95% bootstrap confidence intervals:

{metric_table(metrics_with_ci, "walk_forward")}

Locked out-of-sample metrics with 95% bootstrap confidence intervals:

{metric_table(metrics_with_ci, "out_of_sample")}

On the locked OOS set, the model hit rate is {oos_hit:.2%}, below the majority baseline's {oos_baseline_hit:.2%}. AUC is {oos_auc:.3f}, only slightly above 0.50. The model Sharpe is {oos_sharpe:.3f}, but the confidence interval is very wide and includes strongly negative values. That is an important result, not a footnote: it means the apparent OOS trading performance is too uncertain to claim a real economic edge. The baseline Sharpe is {oos_baseline_sharpe:.3f}, which is close enough that the model does not clearly dominate a naive long-up-class rule.

Feature importance from the final model:

{importance_table}

## 6. Is The Edge Real?

My conclusion is no: there is not enough evidence to claim a real predictive or tradable edge.

First, the OOS directional edge is negative. The model's hit rate is {oos_hit:.2%}, while the majority baseline is {oos_baseline_hit:.2%}. A paired permutation test of model hit-rate edge versus the majority baseline gives p={edge_p_value:.4f}. This does not reject the null that the apparent difference is noise.

Second, the random-label test does not support a robust signal. Keeping the model scores fixed and shuffling OOS labels gives an AUC p-value of {random_auc_p_value:.4f}. A result this close to random labels is not strong evidence of learnable structure.

Third, the train/test boundary sensitivity is unstable. If a small change in the OOS start date changes the conclusion, the result is probably regime- and sample-dependent rather than a durable edge.

{sensitivity_table}

Fourth, the OOS Sharpe ratio looks superficially positive, but its 95% bootstrap interval is wide: -2.0850 to 3.6707. A real edge should survive uncertainty estimates; this one does not. This is especially important because the simple backtest ignores transaction costs, slippage, financing, margin, and operational constraints.

Diagnostics:

{diagnostics_table}

## 7. What Could Still Be Wrong?

Several things could still be wrong even after the leakage audit. The starter features were provided as a convenience and not fully documented, so some rolling calculations may have implementation assumptions that are not visible from the column names. Same-day close-based features are acceptable only if the trading decision is made after the close for the next session; they would be invalid for an intraday signal. The bootstrap intervals treat resampled days as independent, which is imperfect for financial time series. The strategy diagnostics use a simple long/short direction rule and do not include transaction costs. Finally, the sample is short and covers a limited market regime, so the model may be fitting noise.

## 8. What I Would Build Next

The next version should focus less on model complexity and more on validation quality. I would add transaction-cost-aware backtests, block bootstrap confidence intervals, probability calibration, threshold tuning using only walk-forward folds, and a true external test period when more data becomes available. I would also rebuild all features from raw OHLCV instead of relying on precomputed starter columns, then compare a no-trade option for low-confidence days against the always-long/always-short simplification used here.

## Bottom Line

This project produced a reproducible MLflow-tracked research pipeline, but the honest research conclusion is negative. The LightGBM model does not beat the majority baseline on locked OOS hit rate, its walk-forward performance is near random, and its positive OOS Sharpe is too uncertain to defend as evidence of an edge.
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
