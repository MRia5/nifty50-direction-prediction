from __future__ import annotations

from pathlib import Path

import pandas as pd

from financial_ml_project.features import SELECTED_FEATURES


def read_price_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"])
    return data.sort_values("date").reset_index(drop=True)


def make_next_day_direction(nifty: pd.DataFrame) -> pd.DataFrame:
    target = nifty[["date", "Close"]].copy()
    target["next_close"] = target["Close"].shift(-1)
    target["next_return"] = target["next_close"] / target["Close"] - 1
    target["target"] = (target["next_close"] > target["Close"]).astype(float)
    target.loc[target["next_close"].isna(), ["target", "next_return"]] = float("nan")
    return target[["date", "target", "next_return"]]


def load_feature_table(feature_path: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_path, parse_dates=["date"])
    return features.sort_values("date").reset_index(drop=True)


def load_modeling_frame(feature_path: Path, nifty_path: Path) -> pd.DataFrame:
    features = load_feature_table(feature_path)
    missing_features = sorted(set(SELECTED_FEATURES) - set(features.columns))
    if missing_features:
        raise ValueError(f"Missing selected features: {missing_features}")

    nifty = read_price_data(nifty_path)
    target = make_next_day_direction(nifty)

    frame = features[["date", *SELECTED_FEATURES]].merge(target, on="date", how="inner")
    frame = frame.dropna(subset=["target", "next_return"]).reset_index(drop=True)
    frame["target"] = frame["target"].astype(int)
    return frame
