from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from financial_ml_project.config import (
    INITIAL_TRAIN_MONTHS,
    INSAMPLE_END,
    INSAMPLE_START,
    OOS_END,
    OOS_START,
    RANDOM_STATE,
)
from financial_ml_project.features import SELECTED_FEATURES


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_lightgbm_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            (
                "model",
                lgb.LGBMClassifier(
                    objective="binary",
                    boosting_type="gbdt",
                    n_estimators=120,
                    learning_rate=0.03,
                    num_leaves=7,
                    max_depth=3,
                    min_child_samples=25,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                ),
            ),
        ]
    )


def model_params() -> dict[str, float | int | str]:
    model = build_lightgbm_pipeline().named_steps["model"]
    return {f"lightgbm_{key}": value for key, value in model.get_params().items()}


def monthly_expanding_folds(frame: pd.DataFrame) -> list[Fold]:
    in_sample = frame[
        (frame["date"] >= pd.Timestamp(INSAMPLE_START))
        & (frame["date"] <= pd.Timestamp(INSAMPLE_END))
    ].copy()
    months = pd.period_range(
        in_sample["date"].min().to_period("M"),
        in_sample["date"].max().to_period("M"),
        freq="M",
    )

    folds: list[Fold] = []
    for fold_id, test_month in enumerate(months[INITIAL_TRAIN_MONTHS:], start=1):
        train_months = months[: months.get_loc(test_month)]
        train = in_sample[in_sample["date"].dt.to_period("M").isin(train_months)].copy()
        test = in_sample[in_sample["date"].dt.to_period("M") == test_month].copy()
        if train.empty or test.empty:
            continue
        folds.append(
            Fold(
                fold_id=fold_id,
                train=train,
                test=test,
                train_start=train["date"].min(),
                train_end=train["date"].max(),
                test_start=test["date"].min(),
                test_end=test["date"].max(),
            )
        )
    return folds


def out_of_sample_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[
        (frame["date"] >= pd.Timestamp(INSAMPLE_START))
        & (frame["date"] <= pd.Timestamp(INSAMPLE_END))
    ].copy()
    test = frame[
        (frame["date"] >= pd.Timestamp(OOS_START))
        & (frame["date"] <= pd.Timestamp(OOS_END))
    ].copy()
    return train, test


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[SELECTED_FEATURES], frame["target"]


def predict_with_model(model: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    x_test, _ = feature_matrix(test)
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "date": test["date"].values,
            "actual": test["target"].values,
            "predicted": predicted,
            "probability_up": probability,
            "next_return": test["next_return"].values,
        }
    )


def majority_baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    majority_class = int(train["target"].mean() >= 0.5)
    probability = float(train["target"].mean())
    return pd.DataFrame(
        {
            "date": test["date"].values,
            "actual": test["target"].values,
            "predicted": majority_class,
            "probability_up": probability,
            "next_return": test["next_return"].values,
        }
    )


def fit_model(train: pd.DataFrame) -> Pipeline:
    x_train, y_train = feature_matrix(train)
    model = build_lightgbm_pipeline()
    model.fit(x_train, y_train)
    return model


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    values = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": SELECTED_FEATURES, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
