# Financial ML Track Report

## Objective

Predict whether the NIFTY 50 close will rise on the next trading day using daily market, volume, volatility, Bank Nifty, and India VIX features.

## Data

- Total usable observations: 987
- Training observations: 789
- Test observations: 198
- Target: `1` when `close[t + 1] > close[t]`, otherwise `0`

The split is chronological. The final 20 percent of observations are used only for out-of-sample testing.

## Models Compared

| model               |   cv_accuracy_mean |   cv_accuracy_std |
|:--------------------|-------------------:|------------------:|
| logistic_regression |             0.8351 |            0.0258 |
| gradient_boosting   |             0.7939 |            0.0412 |
| random_forest       |             0.7649 |            0.0574 |

Selected model: `logistic_regression`.

## Test Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.8232 |
| Precision | 0.8198 |
| Recall | 0.8585 |
| F1 | 0.8387 |
| ROC-AUC | 0.9134 |
| Actual up-day rate | 0.5354 |
| Predicted up-day rate | 0.5606 |

## Most Important Features

| feature           |   importance |
|:------------------|-------------:|
| ma5_smooth_signal |     5.004155 |
| close_vs_ma5      |     2.090206 |
| ret_overnight     |     0.963359 |
| close_vs_ma50     |     0.627379 |
| ret_intraday      |     0.450470 |
| log_volume        |     0.350904 |
| ret_zscore        |     0.345875 |
| ret_1d            |     0.345875 |
| volume_ratio_20d  |     0.330676 |
| bn_ret_1d         |     0.282154 |

## Interpretation

This is a supervised directional model rather than a trading system. The most useful next step would be to evaluate turnover, transaction costs, position sizing, and drawdowns before treating the signal as tradable.
