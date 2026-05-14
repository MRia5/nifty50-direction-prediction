from __future__ import annotations

import pandas as pd

from financial_ml_project.metrics import (
    compute_metrics,
    directional_accuracy_p_value,
    permutation_edge_test,
    random_label_test,
)
from financial_ml_project.modeling import fit_model, majority_baseline_predictions, predict_with_model


def build_edge_diagnostics(
    oos_predictions: pd.DataFrame,
    oos_baseline: pd.DataFrame,
) -> pd.DataFrame:
    baseline_hit_rate = compute_metrics(oos_baseline)["hit_rate"]
    significance = directional_accuracy_p_value(oos_predictions, baseline_hit_rate)
    permutation = permutation_edge_test(oos_predictions, oos_baseline)
    random_labels = random_label_test(oos_predictions)

    rows = [
        {
            "test": "hit_rate_vs_majority_baseline",
            "statistic": "two_sided_p_value",
            "value": significance,
            "interpretation": "No statistically reliable directional edge if p-value is above 0.05.",
        },
        {
            "test": "paired_permutation_edge",
            "statistic": "observed_hit_rate_edge",
            "value": permutation["observed_hit_rate_edge"],
            "interpretation": "Model hit rate minus majority-baseline hit rate on the same OOS dates.",
        },
        {
            "test": "paired_permutation_edge",
            "statistic": "p_value",
            "value": permutation["p_value"],
            "interpretation": "Probability of an absolute edge this large under paired sign randomization.",
        },
        {
            "test": "random_label_baseline",
            "statistic": "observed_auc",
            "value": random_labels["observed_auc"],
            "interpretation": "Observed model AUC on the true OOS labels.",
        },
        {
            "test": "random_label_baseline",
            "statistic": "random_auc_mean",
            "value": random_labels["random_auc_mean"],
            "interpretation": "Mean AUC after shuffling OOS labels against fixed model scores.",
        },
        {
            "test": "random_label_baseline",
            "statistic": "auc_p_value",
            "value": random_labels["auc_p_value"],
            "interpretation": "Share of random-label trials with AUC at least as high as observed.",
        },
    ]
    return pd.DataFrame(rows)


def split_boundary_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        ("oos_from_2025_06", "2025-05-31", "2025-06-01", "2025-12-31"),
        ("oos_from_2025_07", "2025-06-30", "2025-07-01", "2025-12-31"),
        ("oos_from_2025_08", "2025-07-31", "2025-08-01", "2025-12-31"),
    ]
    rows = []
    for scenario, train_end, test_start, test_end in scenarios:
        train = frame[frame["date"] <= pd.Timestamp(train_end)].copy()
        test = frame[
            (frame["date"] >= pd.Timestamp(test_start))
            & (frame["date"] <= pd.Timestamp(test_end))
        ].copy()
        model = fit_model(train)
        predictions = predict_with_model(model, test)
        baseline = majority_baseline_predictions(train, test)
        model_metrics = compute_metrics(predictions)
        baseline_metrics = compute_metrics(baseline)
        rows.append(
            {
                "scenario": scenario,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "test_rows": len(test),
                "model_auc": model_metrics["auc"],
                "baseline_auc": baseline_metrics["auc"],
                "model_hit_rate": model_metrics["hit_rate"],
                "baseline_hit_rate": baseline_metrics["hit_rate"],
                "model_sharpe": model_metrics["sharpe_ratio"],
                "baseline_sharpe": baseline_metrics["sharpe_ratio"],
            }
        )
    return pd.DataFrame(rows)
