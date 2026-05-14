from __future__ import annotations

from financial_ml_project.config import FEATURE_FILE, NIFTY_FILE, OUTPUT_DIR
from financial_ml_project.data import load_modeling_frame
from financial_ml_project.modeling import (
    build_models,
    choose_model,
    evaluate_model,
    extract_feature_importance,
    split_chronologically,
)
from financial_ml_project.reporting import save_diagnostic_plots, save_metrics, write_report


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_modeling_frame(FEATURE_FILE, NIFTY_FILE)
    split = split_chronologically(frame)
    models = build_models()

    model_name, model, cv_results = choose_model(models, split)
    metrics, predictions, confusion = evaluate_model(model, split)
    feature_importance = extract_feature_importance(model, list(split.x_train.columns))

    save_metrics(metrics, model_name, cv_results, OUTPUT_DIR)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    save_diagnostic_plots(model, split, confusion, OUTPUT_DIR)
    write_report(
        output_dir=OUTPUT_DIR,
        model_name=model_name,
        metrics=metrics,
        cv_results=cv_results,
        feature_importance=feature_importance,
        n_rows=len(frame),
        train_rows=len(split.x_train),
        test_rows=len(split.x_test),
    )

    print(f"Selected model: {model_name}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Report written to: {OUTPUT_DIR / 'report.md'}")
