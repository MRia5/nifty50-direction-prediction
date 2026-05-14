from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd

from financial_ml_project.config import FEATURE_FILE, MLRUNS_DIR, NIFTY_FILE, OUTPUT_DIR
from financial_ml_project.data import load_feature_table, load_modeling_frame
from financial_ml_project.diagnostics import build_edge_diagnostics, split_boundary_sensitivity
from financial_ml_project.features import build_feature_audit
from financial_ml_project.metrics import bootstrap_metric_intervals, compute_metrics
from financial_ml_project.modeling import (
    extract_feature_importance,
    feature_matrix,
    fit_model,
    majority_baseline_predictions,
    model_params,
    monthly_expanding_folds,
    out_of_sample_split,
    predict_with_model,
)
from financial_ml_project.reporting import save_confusion_plot, save_roc_plot, write_report


def add_metadata(metrics: pd.DataFrame, model: str, sample: str) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics.insert(0, "sample", sample)
    metrics.insert(1, "model", model)
    return metrics


def log_training_run(
    run_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    model,
    params: dict[str, object],
    artifact_prefix: str,
) -> None:
    metrics = compute_metrics(predictions)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("train_start", str(train["date"].min().date()))
        mlflow.log_param("train_end", str(train["date"].max().date()))
        mlflow.log_param("test_start", str(test["date"].min().date()))
        mlflow.log_param("test_end", str(test["date"].max().date()))

        prediction_path = OUTPUT_DIR / f"{artifact_prefix}_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        mlflow.log_artifact(str(prediction_path), artifact_path="predictions")
        mlflow.sklearn.log_model(model, name="model")


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("nifty50-direction-prediction")

    raw_features = load_feature_table(FEATURE_FILE)
    feature_audit = build_feature_audit(raw_features)
    feature_audit.to_csv(OUTPUT_DIR / "feature_audit.csv", index=False)

    frame = load_modeling_frame(FEATURE_FILE, NIFTY_FILE)
    folds = monthly_expanding_folds(frame)

    fold_rows = []
    walk_forward_predictions = []
    walk_forward_baseline = []
    params = model_params()

    for fold in folds:
        model = fit_model(fold.train)
        predictions = predict_with_model(model, fold.test)
        baseline = majority_baseline_predictions(fold.train, fold.test)
        fold_metrics = compute_metrics(predictions)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start.date(),
                "train_end": fold.train_end.date(),
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                **fold_metrics,
            }
        )
        predictions["fold_id"] = fold.fold_id
        baseline["fold_id"] = fold.fold_id
        walk_forward_predictions.append(predictions)
        walk_forward_baseline.append(baseline)
        log_training_run(
            run_name=f"walk_forward_fold_{fold.fold_id:02d}",
            train=fold.train,
            test=fold.test,
            predictions=predictions,
            model=model,
            params={**params, "fold_id": fold.fold_id, "run_type": "walk_forward"},
            artifact_prefix=f"fold_{fold.fold_id:02d}",
        )

    train_oos, test_oos = out_of_sample_split(frame)
    final_model = fit_model(train_oos)
    oos_predictions = predict_with_model(final_model, test_oos)
    oos_baseline = majority_baseline_predictions(train_oos, test_oos)
    log_training_run(
        run_name="locked_out_of_sample",
        train=train_oos,
        test=test_oos,
        predictions=oos_predictions,
        model=final_model,
        params={**params, "run_type": "locked_out_of_sample"},
        artifact_prefix="locked_oos",
    )

    walk_forward_predictions_df = pd.concat(walk_forward_predictions, ignore_index=True)
    walk_forward_baseline_df = pd.concat(walk_forward_baseline, ignore_index=True)
    fold_summary = pd.DataFrame(fold_rows)
    feature_importance = extract_feature_importance(final_model)
    edge_diagnostics = build_edge_diagnostics(oos_predictions, oos_baseline)
    boundary_sensitivity = split_boundary_sensitivity(frame)

    metrics_with_ci = pd.concat(
        [
            add_metadata(bootstrap_metric_intervals(walk_forward_predictions_df), "lightgbm", "walk_forward"),
            add_metadata(bootstrap_metric_intervals(walk_forward_baseline_df), "majority_baseline", "walk_forward"),
            add_metadata(bootstrap_metric_intervals(oos_predictions), "lightgbm", "out_of_sample"),
            add_metadata(bootstrap_metric_intervals(oos_baseline), "majority_baseline", "out_of_sample"),
        ],
        ignore_index=True,
    )

    fold_summary.to_csv(OUTPUT_DIR / "walk_forward_folds.csv", index=False)
    walk_forward_predictions_df.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    walk_forward_baseline_df.to_csv(OUTPUT_DIR / "walk_forward_baseline_predictions.csv", index=False)
    oos_predictions.to_csv(OUTPUT_DIR / "oos_predictions.csv", index=False)
    oos_baseline.to_csv(OUTPUT_DIR / "oos_baseline_predictions.csv", index=False)
    metrics_with_ci.to_csv(OUTPUT_DIR / "metrics_with_bootstrap_ci.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    edge_diagnostics.to_csv(OUTPUT_DIR / "edge_diagnostics.csv", index=False)
    boundary_sensitivity.to_csv(OUTPUT_DIR / "split_boundary_sensitivity.csv", index=False)

    save_confusion_plot(oos_predictions, OUTPUT_DIR / "confusion_matrix.png")
    save_roc_plot(oos_predictions, OUTPUT_DIR / "roc_curve.png")
    write_report(
        output_dir=OUTPUT_DIR,
        fold_summary=fold_summary,
        metrics_with_ci=metrics_with_ci,
        feature_audit=feature_audit,
        feature_importance=feature_importance,
        edge_diagnostics=edge_diagnostics,
        boundary_sensitivity=boundary_sensitivity,
        frame=frame,
        n_rows=len(frame),
        n_folds=len(folds),
        oos_rows=len(test_oos),
    )

    print(f"Walk-forward folds: {len(folds)}")
    print(f"OOS rows: {len(test_oos)}")
    print(f"Report written to: {OUTPUT_DIR / 'report.md'}")
