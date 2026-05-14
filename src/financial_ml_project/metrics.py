from __future__ import annotations

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
