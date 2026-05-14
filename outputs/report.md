# Financial ML Track Research Report

## 1. Problem Framing

The task is to predict whether the next NIFTY 50 close is above the current close. I define the label as `1` when `close[t+1] > close[t]` and `0` otherwise. Flat returns are therefore treated as down/non-up days. I did not remove near-zero moves, because removing them would make the target easier after observing the next-day return. In this dataset there are 1 exactly flat next-day returns and 65 near-zero returns with absolute next-day return below 5 basis points.

The unconditional up-day rate in the usable sample is 53.09%, so a majority-class classifier has a full-sample baseline accuracy of 53.09%. In the locked July-December 2025 test period, the majority-class baseline accuracy is 51.20%. This is the minimum hurdle: a useful model should beat this baseline out of sample and should do so by enough that random variation is an implausible explanation.

## 2. Data, Split Design, And Tracking

- Total usable observations: 987
- In-sample walk-forward period: January 2022 through June 2025
- Locked out-of-sample period: July 2025 through December 2025
- Expanding walk-forward folds: 36
- Out-of-sample observations: 125

The validation loop trains on months 1-N, predicts month N+1, shifts forward, and repeats. The July-December 2025 block is locked away until the final evaluation. Every training fold is wrapped in `mlflow.start_run()` from the beginning of the fold, with parameters, metrics, prediction artifacts, and model artifacts written to the committed `mlruns/` directory.

## 3. Feature Audit And Final Features

I used exactly 12 features. The selection rule was conservative: use features that can plausibly be known at today's close, are trailing/same-day rather than forward-looking, and are interpretable enough to defend. I did not find an explicit tomorrow-close column. I still dropped opaque engineered columns where the construction was not auditable.

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

The main suspicious column was `ma5_smooth_signal`. It is not a raw market measurement such as return, volatility, volume, or VIX. It is a prebuilt "signal" column, its exact formula is not documented in the data bundle, and its name suggests a smoothed trading rule rather than a transparent feature. Because the assignment asks for leakage awareness and because the provided starter features were explicitly "not rigorously audited," I treated it as medium leakage/model-design risk and removed it.

Dropped feature audit:

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

## 4. Model

The model is a single `lightgbm.LGBMClassifier`. I removed logistic regression, random forest, and sklearn's `GradientBoostingClassifier` so the experiment is not a model-selection exercise disguised as a final result. Hyperparameters are intentionally modest: shallow trees, small leaves, regularization, and subsampling to reduce overfitting on a short financial time series.

## 5. Results

Walk-forward performance is weak. The model's walk-forward AUC is 0.478, compared with 0.464 for the majority baseline. The model's walk-forward hit rate is 50.34%, while the majority baseline hit rate is 51.15%. In plain English: before the locked test period, the model is not convincingly better than a naive class-prior rule.

Walk-forward fold details:

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

In-sample walk-forward metrics with 95% bootstrap confidence intervals:

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

Locked out-of-sample metrics with 95% bootstrap confidence intervals:

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

On the locked OOS set, the model hit rate is 50.40%, below the majority baseline's 51.20%. AUC is 0.524, only slightly above 0.50. The model Sharpe is 0.768, but the confidence interval is very wide and includes strongly negative values. That is an important result, not a footnote: it means the apparent OOS trading performance is too uncertain to claim a real economic edge. The baseline Sharpe is 0.616, which is close enough that the model does not clearly dominate a naive long-up-class rule.

Feature importance from the final model:

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

## 6. Is The Edge Real?

My conclusion is no: there is not enough evidence to claim a real predictive or tradable edge.

First, the OOS directional edge is negative. The model's hit rate is 50.40%, while the majority baseline is 51.20%. A paired permutation test of model hit-rate edge versus the majority baseline gives p=1.0000. This does not reject the null that the apparent difference is noise.

Second, the random-label test does not support a robust signal. Keeping the model scores fixed and shuffling OOS labels gives an AUC p-value of 0.3246. A result this close to random labels is not strong evidence of learnable structure.

Third, the train/test boundary sensitivity is unstable. If a small change in the OOS start date changes the conclusion, the result is probably regime- and sample-dependent rather than a durable edge.

| scenario         | train_end   | test_start   | test_end   |   test_rows |   model_auc |   baseline_auc |   model_hit_rate |   baseline_hit_rate |   model_sharpe |   baseline_sharpe |
|:-----------------|:------------|:-------------|:-----------|------------:|------------:|---------------:|-----------------:|--------------------:|---------------:|------------------:|
| oos_from_2025_06 | 2025-05-31  | 2025-06-01   | 2025-12-31 |         146 |      0.4978 |         0.5000 |           0.5068 |              0.5274 |         0.1135 |            1.1912 |
| oos_from_2025_07 | 2025-06-30  | 2025-07-01   | 2025-12-31 |         125 |      0.5239 |         0.5000 |           0.5040 |              0.5120 |         0.7680 |            0.6158 |
| oos_from_2025_08 | 2025-07-31  | 2025-08-01   | 2025-12-31 |         102 |      0.5412 |         0.5000 |           0.5392 |              0.5392 |         1.6862 |            1.9520 |

Fourth, the OOS Sharpe ratio looks superficially positive, but its 95% bootstrap interval is wide: -2.0850 to 3.6707. A real edge should survive uncertainty estimates; this one does not. This is especially important because the simple backtest ignores transaction costs, slippage, financing, margin, and operational constraints.

Diagnostics:

| test                          | statistic              |   value | interpretation                                                              |
|:------------------------------|:-----------------------|--------:|:----------------------------------------------------------------------------|
| hit_rate_vs_majority_baseline | two_sided_p_value      |  0.8580 | No statistically reliable directional edge if p-value is above 0.05.        |
| paired_permutation_edge       | observed_hit_rate_edge | -0.0080 | Model hit rate minus majority-baseline hit rate on the same OOS dates.      |
| paired_permutation_edge       | p_value                |  1.0000 | Probability of an absolute edge this large under paired sign randomization. |
| random_label_baseline         | observed_auc           |  0.5239 | Observed model AUC on the true OOS labels.                                  |
| random_label_baseline         | random_auc_mean        |  0.4999 | Mean AUC after shuffling OOS labels against fixed model scores.             |
| random_label_baseline         | auc_p_value            |  0.3246 | Share of random-label trials with AUC at least as high as observed.         |

## 7. What Could Still Be Wrong?

Several things could still be wrong even after the leakage audit. The starter features were provided as a convenience and not fully documented, so some rolling calculations may have implementation assumptions that are not visible from the column names. Same-day close-based features are acceptable only if the trading decision is made after the close for the next session; they would be invalid for an intraday signal. The bootstrap intervals treat resampled days as independent, which is imperfect for financial time series. The strategy diagnostics use a simple long/short direction rule and do not include transaction costs. Finally, the sample is short and covers a limited market regime, so the model may be fitting noise.

## 8. What I Would Build Next

The next version should focus less on model complexity and more on validation quality. I would add transaction-cost-aware backtests, block bootstrap confidence intervals, probability calibration, threshold tuning using only walk-forward folds, and a true external test period when more data becomes available. I would also rebuild all features from raw OHLCV instead of relying on precomputed starter columns, then compare a no-trade option for low-confidence days against the always-long/always-short simplification used here.

## Bottom Line

This project produced a reproducible MLflow-tracked research pipeline, but the honest research conclusion is negative. The LightGBM model does not beat the majority baseline on locked OOS hit rate, its walk-forward performance is near random, and its positive OOS Sharpe is too uncertain to defend as evidence of an edge.
