from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from financial_ml_project.config import BOOTSTRAP_SAMPLES, RANDOM_STATE


METRIC_COLUMNS = [
    "auc",
    "balanced_accuracy",
    "hit_rate",
    "sharpe_ratio",
    "max_drawdown",
    "turnover",
    "tn",
    "fp",
    "fn",
    "tp",
]


def strategy_returns(predictions: pd.DataFrame) -> np.ndarray:
    signal = np.where(predictions["predicted"].to_numpy() == 1, 1.0, -1.0)
    return signal * predictions["next_return"].to_numpy()


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return float("nan")
    curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(curve)
    drawdown = curve / running_max - 1
    return float(drawdown.min())


def sharpe_ratio(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.isclose(np.std(returns, ddof=1), 0):
        return 0.0
    return float(np.sqrt(252) * np.mean(returns) / np.std(returns, ddof=1))


def turnover(predictions: pd.DataFrame) -> float:
    signal = np.where(predictions["predicted"].to_numpy() == 1, 1, -1)
    if len(signal) < 2:
        return 0.0
    return float(np.abs(np.diff(signal)).sum() / (2 * (len(signal) - 1)))


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["actual"].to_numpy()
    y_pred = predictions["predicted"].to_numpy()
    probability = predictions["probability_up"].to_numpy()
    returns = strategy_returns(predictions)

    if len(np.unique(y_true)) == 2 and len(np.unique(probability)) > 1:
        auc = float(roc_auc_score(y_true, probability))
    else:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "hit_rate": float(np.mean(y_true == y_pred)),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "turnover": turnover(predictions),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def bootstrap_metric_intervals(
    predictions: pd.DataFrame,
    n_samples: int = BOOTSTRAP_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    values = {metric: [] for metric in METRIC_COLUMNS}

    for _ in range(n_samples):
        sample_index = rng.integers(0, len(predictions), len(predictions))
        sample = predictions.iloc[sample_index].reset_index(drop=True)
        sample_metrics = compute_metrics(sample)
        for metric in METRIC_COLUMNS:
            values[metric].append(sample_metrics[metric])

    point_metrics = compute_metrics(predictions)
    for metric in METRIC_COLUMNS:
        distribution = np.array(values[metric], dtype=float)
        ci_low = float(np.nanpercentile(distribution, 2.5))
        ci_high = float(np.nanpercentile(distribution, 97.5))
        point = point_metrics[metric]
        rows.append(
            {
                "metric": metric,
                "value": point,
                "ci_low": min(ci_low, point),
                "ci_high": max(ci_high, point),
            }
        )
    return pd.DataFrame(rows)


def directional_accuracy_p_value(predictions: pd.DataFrame, null_accuracy: float) -> float:
    """Two-sided normal-approximation p-value for hit rate versus a baseline hit rate."""
    hits = (predictions["actual"].to_numpy() == predictions["predicted"].to_numpy()).astype(float)
    observed = hits.mean()
    n = len(hits)
    standard_error = np.sqrt(null_accuracy * (1 - null_accuracy) / n)
    if np.isclose(standard_error, 0):
        return 1.0
    z_score = abs(observed - null_accuracy) / standard_error
    return float(2 * (1 - _normal_cdf(z_score)))


def permutation_edge_test(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    n_permutations: int = 5000,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Paired permutation test for model hit-rate edge over the baseline."""
    model_hit = (model_predictions["actual"].to_numpy() == model_predictions["predicted"].to_numpy()).astype(float)
    baseline_hit = (
        baseline_predictions["actual"].to_numpy() == baseline_predictions["predicted"].to_numpy()
    ).astype(float)
    paired_diff = model_hit - baseline_hit
    observed_edge = float(paired_diff.mean())
    rng = np.random.default_rng(random_state)
    null_edges = []
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(paired_diff))
        null_edges.append(float(np.mean(paired_diff * signs)))
    null_edges_array = np.array(null_edges)
    p_value = float(np.mean(np.abs(null_edges_array) >= abs(observed_edge)))
    return {
        "observed_hit_rate_edge": observed_edge,
        "p_value": p_value,
        "null_edge_mean": float(null_edges_array.mean()),
        "null_edge_95_low": float(np.percentile(null_edges_array, 2.5)),
        "null_edge_95_high": float(np.percentile(null_edges_array, 97.5)),
    }


def random_label_test(
    predictions: pd.DataFrame,
    n_permutations: int = 5000,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Shuffle OOS labels against fixed model scores to test whether results beat random labels."""
    rng = np.random.default_rng(random_state)
    actual = predictions["actual"].to_numpy()
    predicted = predictions["predicted"].to_numpy()
    probability = predictions["probability_up"].to_numpy()
    observed_hit_rate = float(np.mean(actual == predicted))
    observed_auc = float(roc_auc_score(actual, probability))

    random_hit_rates = []
    random_aucs = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(actual)
        random_hit_rates.append(float(np.mean(shuffled == predicted)))
        if len(np.unique(shuffled)) == 2:
            random_aucs.append(float(roc_auc_score(shuffled, probability)))

    hit_rate_array = np.array(random_hit_rates)
    auc_array = np.array(random_aucs)
    return {
        "observed_hit_rate": observed_hit_rate,
        "random_hit_rate_mean": float(hit_rate_array.mean()),
        "random_hit_rate_95_low": float(np.percentile(hit_rate_array, 2.5)),
        "random_hit_rate_95_high": float(np.percentile(hit_rate_array, 97.5)),
        "hit_rate_p_value": float(np.mean(hit_rate_array >= observed_hit_rate)),
        "observed_auc": observed_auc,
        "random_auc_mean": float(auc_array.mean()),
        "random_auc_95_low": float(np.percentile(auc_array, 2.5)),
        "random_auc_95_high": float(np.percentile(auc_array, 97.5)),
        "auc_p_value": float(np.mean(auc_array >= observed_auc)),
    }


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1 + math.erf(value / np.sqrt(2))))
