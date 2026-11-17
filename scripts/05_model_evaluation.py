"""LOBO error analysis and robustness check (tuned Random Forest).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config
from src import evaluation as ev


def main():
    config.ensure_directories()
    prediction_file = (config.RF_DIR
                       / "random_forest_baseline_tuned_lobo_predictions.csv")
    pred_df = ev.load_tuned_predictions(prediction_file)
    print(f"Prediction file loaded: {prediction_file}")
    print(f"Tuned RF prediction rows: {len(pred_df)}")
    print(f"Batteries: {sorted(pred_df['battery_id'].unique())}")
    overall_error = ev.overall_error_statistics(pred_df)
    battery_error = ev.error_by_battery(pred_df)
    soh_range_error = ev.error_by_soh_range(pred_df)
    threshold_results = ev.accuracy_thresholds(pred_df)
    largest = ev.largest_errors(pred_df)

    print("\nOverall LOBO error statistics:")
    print(overall_error.to_string(index=False))
    print("\nError statistics by battery:")
    print(battery_error.to_string(index=False))
    print("\nError statistics by SoH range:")
    print(soh_range_error.to_string(index=False))
    print("\nPrediction accuracy thresholds:")
    print(threshold_results.to_string(index=False))
    print(f"\n{config.N_LARGEST_ERRORS} largest LOBO prediction errors:")
    print(largest[["battery_id", "actual_soh", "predicted_soh",
                   "error", "absolute_error"]].to_string(index=False))

    # Plots
    ev.plot_residuals_vs_soh(
        pred_df, config.EVAL_DIR / "random_forest_lobo_residuals_vs_soh.png")
    ev.plot_absolute_error_vs_soh(
        pred_df, config.EVAL_DIR / "random_forest_lobo_absolute_error_vs_soh.png")
    ev.plot_mae_by_battery(
        battery_error, config.EVAL_DIR / "random_forest_lobo_mae_by_battery.png")

    # Tables
    overall_error.to_csv(
        config.EVAL_DIR / "random_forest_lobo_error_statistics.csv", index=False)
    battery_error.to_csv(
        config.EVAL_DIR / "random_forest_lobo_error_by_battery.csv", index=False)
    soh_range_error.to_csv(
        config.EVAL_DIR / "random_forest_lobo_error_by_soh_range.csv",
        index=False)
    threshold_results.to_csv(
        config.EVAL_DIR / "random_forest_lobo_accuracy_thresholds.csv",
        index=False)
    largest.to_csv(
        config.EVAL_DIR / "random_forest_lobo_largest_errors.csv", index=False)
    pred_df.to_csv(
        config.EVAL_DIR / "random_forest_lobo_predictions_with_errors.csv",
        index=False)
    print(f"\nOverall mean error : {pred_df['error'].mean():.6f}")
    print(f"Overall MAE        : {pred_df['absolute_error'].mean():.6f}")
    print(f"Overall RMSE       : {np.sqrt(pred_df['squared_error'].mean()):.6f}")
    print("\nPrediction accuracy thresholds:")
    for _, row in threshold_results.iterrows():
        print(f"{row['threshold']}: {row['percentage']:.2f}%")
    print(f"\nEvaluation outputs saved to: {config.EVAL_DIR}")


if __name__ == "__main__":
    main()
