from __future__ import annotations

from pathlib import Path

import pandas as pd


MAX_MISSING_RATE = 0.25


def read_price_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)
    return data


def make_next_day_direction(nifty: pd.DataFrame) -> pd.DataFrame:
    target = nifty[["date", "Close"]].copy()
    target["next_close"] = target["Close"].shift(-1)
    target["target"] = (target["next_close"] > target["Close"]).astype(float)
    target.loc[target["next_close"].isna(), "target"] = float("nan")
    return target[["date", "target"]]


def load_modeling_frame(feature_path: Path, nifty_path: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_path, parse_dates=["date"])
    features = features.sort_values("date").reset_index(drop=True)

    nifty = read_price_data(nifty_path)
    target = make_next_day_direction(nifty)

    frame = features.merge(target, on="date", how="inner")
    frame = frame.dropna(subset=["target"]).reset_index(drop=True)
    empty_feature_columns = [
        column
        for column in frame.columns
        if column not in {"date", "target"} and frame[column].isna().all()
    ]
    sparse_feature_columns = [
        column
        for column in frame.columns
        if column not in {"date", "target"}
        and frame[column].isna().mean() > MAX_MISSING_RATE
    ]
    frame = frame.drop(columns=sorted(set(empty_feature_columns + sparse_feature_columns)))
    frame["target"] = frame["target"].astype(int)
    return frame
