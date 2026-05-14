from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from financial_ml_project.config import RANDOM_STATE, TEST_SIZE


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_dates: pd.Series


def split_chronologically(frame: pd.DataFrame, test_size: float = TEST_SIZE) -> SplitData:
    feature_columns = [column for column in frame.columns if column not in {"date", "target"}]
    split_index = int(len(frame) * (1 - test_size))

    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()

    return SplitData(
        x_train=train[feature_columns],
        x_test=test[feature_columns],
        y_train=train["target"],
        y_test=test["target"],
        test_dates=test["date"],
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.04,
                        max_depth=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def choose_model(models: dict[str, Pipeline], split: SplitData) -> tuple[str, Pipeline, pd.DataFrame]:
    cv = TimeSeriesSplit(n_splits=5)
    rows = []

    for name, model in models.items():
        scores = cross_val_score(model, split.x_train, split.y_train, cv=cv, scoring="accuracy")
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": scores.mean(),
                "cv_accuracy_std": scores.std(),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False)
    best_name = str(results.iloc[0]["model"])
    best_model = models[best_name]
    best_model.fit(split.x_train, split.y_train)

    return best_name, best_model, results


def evaluate_model(model: Pipeline, split: SplitData) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    prediction = model.predict(split.x_test)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(split.x_test)[:, 1]
    else:
        probability = prediction.astype(float)

    metrics = {
        "accuracy": accuracy_score(split.y_test, prediction),
        "precision": precision_score(split.y_test, prediction, zero_division=0),
        "recall": recall_score(split.y_test, prediction, zero_division=0),
        "f1": f1_score(split.y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(split.y_test, probability),
        "positive_rate_actual": float(split.y_test.mean()),
        "positive_rate_predicted": float(prediction.mean()),
    }

    predictions = pd.DataFrame(
        {
            "date": split.test_dates.values,
            "actual": split.y_test.values,
            "predicted": prediction,
            "probability_up": probability,
        }
    )

    return metrics, predictions, confusion_matrix(split.y_test, prediction)


def extract_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    fitted_model = model.named_steps["model"]

    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(fitted_model.coef_[0])
    else:
        values = np.zeros(len(feature_names))

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
