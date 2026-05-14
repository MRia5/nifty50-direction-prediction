from __future__ import annotations

import pandas as pd


SELECTED_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_intraday",
    "ret_overnight",
    "high_low_range",
    "volume_ratio_20d",
    "close_vs_ma20",
    "vol_20d",
    "rsi_14",
    "bn_ret_5d",
    "vix_change",
]

FEATURE_JUSTIFICATIONS = {
    "ret_1d": "Captures immediate NIFTY momentum/reversal using only today's close versus yesterday.",
    "ret_5d": "Measures one-week index momentum while staying fully trailing.",
    "ret_20d": "Represents roughly one-month trend pressure available at today's close.",
    "ret_intraday": "Separates same-session open-to-close demand from overnight movement.",
    "ret_overnight": "Captures gap risk and overnight sentiment visible by today's close.",
    "high_low_range": "Proxy for same-day realized volatility and uncertainty.",
    "volume_ratio_20d": "Compares current participation with trailing liquidity conditions.",
    "close_vs_ma20": "Shows current price extension versus a trailing one-month moving average.",
    "vol_20d": "Measures trailing one-month realized volatility of NIFTY returns.",
    "rsi_14": "Standard trailing momentum oscillator for overbought/oversold behavior.",
    "bn_ret_5d": "Adds related-sector momentum from Bank Nifty over the same recent window.",
    "vix_change": "Captures same-day change in implied volatility/risk appetite.",
}

AUDIT_NOTES = {
    "ret_1d": "Selected; simple trailing return, no forward target reference.",
    "ret_5d": "Selected; trailing five-day return, no forward target reference.",
    "ret_10d": "Dropped; overlaps strongly with selected trailing return horizons.",
    "ret_20d": "Selected; trailing 20-day return, no forward target reference.",
    "ret_intraday": "Selected; uses same-day open and close known after market close.",
    "ret_overnight": "Selected; uses current open versus previous close.",
    "high_low_range": "Selected; same-day high-low range known after market close.",
    "log_volume": "Dropped; volume_ratio_20d gives a more normalized liquidity signal.",
    "volume_ratio_20d": "Selected; current volume relative to trailing 20-day history.",
    "close_vs_ma5": "Dropped; overlaps with close_vs_ma20 and short return features.",
    "close_vs_ma20": "Selected; trailing MA distance without future values.",
    "close_vs_ma50": "Dropped; higher missingness and overlaps with 20-day trend.",
    "momentum_5_20": "Dropped; derived spread overlaps selected return horizons.",
    "vol_5d": "Dropped; overlaps with selected vol_20d and short return features.",
    "vol_20d": "Selected; trailing realized volatility with modest missingness.",
    "vol_50d": "Dropped; higher missingness and overlaps with vol_20d.",
    "rsi_14": "Selected; trailing oscillator, no direct future target reference.",
    "bn_ret_1d": "Dropped; bn_ret_5d is less noisy and still trailing.",
    "bn_ret_5d": "Selected; trailing cross-asset momentum from Bank Nifty.",
    "nifty_bn_spread": "Dropped; opaque spread definition and high overlap with NIFTY/Bank Nifty returns.",
    "nifty_bn_corr_20d": "Dropped; rolling correlation is trailing but less directly interpretable.",
    "vix_level": "Dropped; vix_change captures risk regime movement with better stationarity.",
    "vix_change": "Selected; same-day implied-volatility change.",
    "vix_5d_change": "Dropped; overlaps with selected VIX change.",
    "vix_ma_ratio": "Dropped; derived VIX trend overlaps selected VIX change.",
    "close_vs_252d_high": "Dropped; more than 25 percent missing because it needs a 252-day lookback.",
    "close_vs_252d_low": "Dropped; more than 25 percent missing because it needs a 252-day lookback.",
    "dow": "Dropped; calendar control has low market-specific content for the 12-feature limit.",
    "ret_zscore": "Dropped; opaque standardized return may duplicate return/volatility features.",
    "ma5_smooth_signal": "Dropped as suspicious; prebuilt signal-like feature could encode unreviewed label logic.",
    "volume_normalized": "Dropped; volume_ratio_20d is the clearer volume normalization.",
}


def build_feature_audit(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in features.columns:
        if column == "date":
            continue
        missing_rate = float(features[column].isna().mean())
        selected = column in SELECTED_FEATURES
        rows.append(
            {
                "feature": column,
                "selected": selected,
                "missing_rate": missing_rate,
                "leakage_risk": "medium" if column == "ma5_smooth_signal" else "low",
                "decision": "keep" if selected else "drop",
                "reason": AUDIT_NOTES.get(column, "Dropped; not part of the final 12-feature specification."),
                "final_feature_justification": FEATURE_JUSTIFICATIONS.get(column, ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["selected", "feature"], ascending=[False, True])
