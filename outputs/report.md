# Financial ML Track Report

## Objective

Predict next-day NIFTY 50 close direction with a single gradient-boosting model and MLflow-tracked expanding-window validation.

## Data And Splits

- Total usable observations: 987
- Walk-forward in-sample period: January 2022 through June 2025
- Locked out-of-sample period: July 2025 through December 2025
- Expanding walk-forward folds: 36
- Out-of-sample observations: 125

The model trains on months 1-N, predicts month N+1, shifts forward, and repeats. The July-December 2025 block is evaluated once at the end only.

## Model

Only one model is used: `lightgbm.LGBMClassifier`. Logistic regression, random forest, and sklearn's GradientBoostingClassifier were removed. Every walk-forward fold and the final out-of-sample fit are wrapped in `mlflow.start_run()` and logged under the committed `mlruns/` directory.

## Final 12 Features

| feature          | justification                                                                         |
|:-----------------|:--------------------------------------------------------------------------------------|
| ret_1d           | Captures immediate NIFTY momentum/reversal using only today's close versus yesterday. |
| ret_5d           | Measures one-week index momentum while staying fully trailing.                        |
| ret_20d          | Represents roughly one-month trend pressure available at today's close.               |
| ret_intraday     | Separates same-session open-to-close demand from overnight movement.                  |
| ret_overnight    | Captures gap risk and overnight sentiment visible by today's close.                   |
| high_low_range   | Proxy for same-day realized volatility and uncertainty.                               |
| volume_ratio_20d | Compares current participation with trailing liquidity conditions.                    |
| close_vs_ma20    | Shows current price extension versus a trailing one-month moving average.             |
| vol_20d          | Measures trailing one-month realized volatility of NIFTY returns.                     |
| rsi_14           | Standard trailing momentum oscillator for overbought/oversold behavior.               |
| bn_ret_5d        | Adds related-sector momentum from Bank Nifty over the same recent window.             |
| vix_change       | Captures same-day change in implied volatility/risk appetite.                         |

## Feature Audit

Every column in `starter_features.csv` was reviewed. No direct tomorrow-close or future-target feature name was found. Opaque prebuilt signal columns and sparse/overlapping columns were dropped.

| feature            | leakage_risk   | reason                                                                                   |
|:-------------------|:---------------|:-----------------------------------------------------------------------------------------|
| bn_ret_1d          | low            | Dropped; bn_ret_5d is less noisy and still trailing.                                     |
| close_vs_252d_high | low            | Dropped; more than 25 percent missing because it needs a 252-day lookback.               |
| close_vs_252d_low  | low            | Dropped; more than 25 percent missing because it needs a 252-day lookback.               |
| close_vs_ma5       | low            | Dropped; overlaps with close_vs_ma20 and short return features.                          |
| close_vs_ma50      | low            | Dropped; higher missingness and overlaps with 20-day trend.                              |
| dow                | low            | Dropped; calendar control has low market-specific content for the 12-feature limit.      |
| log_volume         | low            | Dropped; volume_ratio_20d gives a more normalized liquidity signal.                      |
| ma5_smooth_signal  | medium         | Dropped as suspicious; prebuilt signal-like feature could encode unreviewed label logic. |
| momentum_5_20      | low            | Dropped; derived spread overlaps selected return horizons.                               |
| nifty_bn_corr_20d  | low            | Dropped; rolling correlation is trailing but less directly interpretable.                |
| nifty_bn_spread    | low            | Dropped; opaque spread definition and high overlap with NIFTY/Bank Nifty returns.        |
| ret_10d            | low            | Dropped; overlaps strongly with selected trailing return horizons.                       |
| ret_zscore         | low            | Dropped; opaque standardized return may duplicate return/volatility features.            |
| vix_5d_change      | low            | Dropped; overlaps with selected VIX change.                                              |
| vix_level          | low            | Dropped; vix_change captures risk regime movement with better stationarity.              |
| vix_ma_ratio       | low            | Dropped; derived VIX trend overlaps selected VIX change.                                 |
| vol_50d            | low            | Dropped; higher missingness and overlaps with vol_20d.                                   |
| vol_5d             | low            | Dropped; overlaps with selected vol_20d and short return features.                       |
| volume_normalized  | low            | Dropped; volume_ratio_20d is the clearer volume normalization.                           |

## Walk-Forward Fold Metrics

|   fold_id | train_start   | train_end   | test_start   | test_end   |   hit_rate |    auc |
|----------:|:--------------|:------------|:-------------|:-----------|-----------:|-------:|
|         1 | 2022-01-03    | 2022-06-30  | 2022-07-01   | 2022-07-29 |     0.2857 | 0.4388 |
|         2 | 2022-01-03    | 2022-07-29  | 2022-08-01   | 2022-08-30 |     0.5500 | 0.4762 |
|         3 | 2022-01-03    | 2022-08-30  | 2022-09-01   | 2022-09-30 |     0.5000 | 0.4643 |
|         4 | 2022-01-03    | 2022-09-30  | 2022-10-03   | 2022-10-31 |     0.5263 | 0.4857 |
|         5 | 2022-01-03    | 2022-10-31  | 2022-11-01   | 2022-11-30 |     0.5714 | 0.5000 |
|         6 | 2022-01-03    | 2022-11-30  | 2022-12-01   | 2022-12-30 |     0.4091 | 0.4083 |
|         7 | 2022-01-03    | 2022-12-30  | 2023-01-02   | 2023-01-31 |     0.4286 | 0.4808 |
|         8 | 2022-01-03    | 2023-01-31  | 2023-02-01   | 2023-02-28 |     0.5000 | 0.5604 |
|         9 | 2022-01-03    | 2023-02-28  | 2023-03-01   | 2023-03-31 |     0.4286 | 0.5091 |
|        10 | 2022-01-03    | 2023-03-31  | 2023-04-03   | 2023-04-28 |     0.5882 | 0.6731 |
|        11 | 2022-01-03    | 2023-04-28  | 2023-05-02   | 2023-05-31 |     0.4091 | 0.2308 |
|        12 | 2022-01-03    | 2023-05-31  | 2023-06-01   | 2023-06-30 |     0.5238 | 0.3556 |
|        13 | 2022-01-03    | 2023-06-30  | 2023-07-03   | 2023-07-31 |     0.7619 | 0.5204 |
|        14 | 2022-01-03    | 2023-07-31  | 2023-08-01   | 2023-08-31 |     0.5455 | 0.4833 |
|        15 | 2022-01-03    | 2023-08-31  | 2023-09-01   | 2023-09-29 |     0.6000 | 0.2812 |
|        16 | 2022-01-03    | 2023-09-29  | 2023-10-03   | 2023-10-31 |     0.5500 | 0.7143 |
|        17 | 2022-01-03    | 2023-10-31  | 2023-11-01   | 2023-11-30 |     0.6000 | 0.6190 |
|        18 | 2022-01-03    | 2023-11-30  | 2023-12-01   | 2023-12-29 |     0.7000 | 0.6400 |
|        19 | 2022-01-03    | 2023-12-29  | 2024-01-01   | 2024-01-31 |     0.3810 | 0.3364 |
|        20 | 2022-01-03    | 2024-01-31  | 2024-02-01   | 2024-02-29 |     0.3810 | 0.2653 |
|        21 | 2022-01-03    | 2024-02-29  | 2024-03-01   | 2024-03-28 |     0.7222 | 0.6528 |
|        22 | 2022-01-03    | 2024-03-28  | 2024-04-01   | 2024-04-30 |     0.4500 | 0.3900 |
|        23 | 2022-01-03    | 2024-04-30  | 2024-05-02   | 2024-05-31 |     0.4286 | 0.4545 |
|        24 | 2022-01-03    | 2024-05-31  | 2024-06-03   | 2024-06-28 |     0.4737 | 0.3714 |
|        25 | 2022-01-03    | 2024-06-28  | 2024-07-01   | 2024-07-31 |     0.5909 | 0.2137 |
|        26 | 2022-01-03    | 2024-07-31  | 2024-08-01   | 2024-08-30 |     0.5238 | 0.5000 |
|        27 | 2022-01-03    | 2024-08-30  | 2024-09-02   | 2024-09-30 |     0.5714 | 0.4352 |
|        28 | 2022-01-03    | 2024-09-30  | 2024-10-01   | 2024-10-31 |     0.3636 | 0.4667 |
|        29 | 2022-01-03    | 2024-10-31  | 2024-11-01   | 2024-11-29 |     0.3684 | 0.5114 |
|        30 | 2022-01-03    | 2024-11-29  | 2024-12-02   | 2024-12-31 |     0.6190 | 0.6944 |
|        31 | 2022-01-03    | 2024-12-31  | 2025-01-01   | 2025-01-31 |     0.5652 | 0.5833 |
|        32 | 2022-01-03    | 2025-01-31  | 2025-02-01   | 2025-02-28 |     0.2000 | 0.1944 |
|        33 | 2022-01-03    | 2025-02-28  | 2025-03-03   | 2025-03-28 |     0.6842 | 0.6786 |
|        34 | 2022-01-03    | 2025-03-28  | 2025-04-01   | 2025-04-30 |     0.4737 | 0.5952 |
|        35 | 2022-01-03    | 2025-04-30  | 2025-05-02   | 2025-05-30 |     0.4762 | 0.4630 |
|        36 | 2022-01-03    | 2025-05-30  | 2025-06-02   | 2025-06-30 |     0.4286 | 0.4135 |

## In-Sample Walk-Forward Metrics With 95% Bootstrap CI

| metric            | lightgbm                      | majority_baseline             |
|:------------------|:------------------------------|:------------------------------|
| auc               | 0.4776 [0.4428, 0.5194]       | 0.4644 [0.4254, 0.5100]       |
| balanced_accuracy | 0.4846 [0.4543, 0.5158]       | 0.4843 [0.4568, 0.5172]       |
| fn                | 139.0000 [120.0000, 159.0000] | 107.0000 [88.0000, 126.0000]  |
| fp                | 228.0000 [201.0000, 252.0000] | 254.0000 [229.0000, 282.0000] |
| hit_rate          | 0.5034 [0.4723, 0.5386]       | 0.5115 [0.4777, 0.5501]       |
| max_drawdown      | -0.1245 [-0.3895, -0.1101]    | -0.2639 [-0.3931, -0.1020]    |
| sharpe_ratio      | 0.0508 [-0.9980, 1.0804]      | 0.1196 [-0.9422, 1.3560]      |
| tn                | 102.0000 [85.4750, 119.0000]  | 76.0000 [61.0000, 93.0000]    |
| tp                | 270.0000 [243.4750, 295.5250] | 302.0000 [276.0000, 328.0000] |
| turnover          | 0.2940 [0.2940, 0.4804]       | 0.0041 [0.0041, 0.4140]       |

## Locked Out-Of-Sample Metrics With 95% Bootstrap CI

| metric            | lightgbm                   | majority_baseline          |
|:------------------|:---------------------------|:---------------------------|
| auc               | 0.5239 [0.4192, 0.6276]    | 0.5000 [0.5000, 0.5000]    |
| balanced_accuracy | 0.4949 [0.4353, 0.5569]    | 0.5000 [0.5000, 0.5000]    |
| fn                | 8.0000 [3.0000, 14.0000]   | 0.0000 [0.0000, 0.0000]    |
| fp                | 54.0000 [43.0000, 66.5250] | 61.0000 [50.0000, 73.0000] |
| hit_rate          | 0.5040 [0.4080, 0.6000]    | 0.5120 [0.4160, 0.6000]    |
| max_drawdown      | -0.0492 [-0.1022, -0.0231] | -0.0454 [-0.1058, -0.0239] |
| sharpe_ratio      | 0.7680 [-2.0850, 3.6707]   | 0.6158 [-2.3653, 3.6332]   |
| tn                | 7.0000 [2.0000, 12.0000]   | 0.0000 [0.0000, 0.0000]    |
| tp                | 56.0000 [46.0000, 68.0000] | 64.0000 [52.0000, 75.0000] |
| turnover          | 0.1774 [0.1248, 0.3065]    | 0.0000 [0.0000, 0.0000]    |

## Feature Importance

| feature          |   importance |
|:-----------------|-------------:|
| vol_20d          |           97 |
| ret_overnight    |           76 |
| close_vs_ma20    |           66 |
| volume_ratio_20d |           42 |
| ret_1d           |           40 |
| vix_change       |           39 |
| high_low_range   |           35 |
| ret_intraday     |           33 |
| bn_ret_5d        |           29 |
| ret_20d          |           25 |
| rsi_14           |           15 |
| ret_5d           |           10 |

## Interpretation

The majority-class baseline is included beside the model for every reported metric. Trading diagnostics use a simple directional strategy: predicted up is long NIFTY for the next day, predicted down is short NIFTY for the next day. Sharpe, drawdown, and turnover therefore describe signal behavior, not a production trading strategy with costs or constraints.
