"""LOBO error analysis and robustness checks for the tuned Random Forest."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import config
from src.model_utils import save_figure


def assign_soh_range(soh):
    if soh >= config.SOH_HIGH_THRESHOLD:
        return "High_SoH"
    if soh >= config.SOH_MEDIUM_THRESHOLD:
        return "Medium_SoH"
    return "Low_SoH"


def load_tuned_predictions(prediction_file, model_name=config.TUNED_RF_NAME):
    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Required prediction file not found:\n{prediction_file}\n\n"
            "Please run scripts/02_RandomForest.py first."
        )
    pred_df = pd.read_csv(prediction_file)
    if "model" not in pred_df.columns:
        raise ValueError("The prediction file does not contain a 'model' column.")

    if model_name not in pred_df["model"].unique():
        raise ValueError(
            f"'{model_name}' was not found.\n"
            f"Available models: {sorted(pred_df['model'].unique())}"
        )

    pred_df = pred_df[pred_df["model"] == model_name].copy()
    required = ["battery_id", "actual_soh", "predicted_soh"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pred_df["error"] = pred_df["predicted_soh"] - pred_df["actual_soh"]
    pred_df["absolute_error"] = pred_df["error"].abs()
    pred_df["squared_error"] = pred_df["error"] ** 2
    pred_df["soh_range"] = pred_df["actual_soh"].apply(assign_soh_range)
    return pred_df


def overall_error_statistics(pred_df):
    return pd.DataFrame({
        "metric": [
            "Mean error", "Mean absolute error", "RMSE",
            "Maximum absolute error", "Minimum error", "Maximum error",
            "Std error",
        ],
        "value": [
            pred_df["error"].mean(),
            pred_df["absolute_error"].mean(),
            np.sqrt(pred_df["squared_error"].mean()),
            pred_df["absolute_error"].max(),
            pred_df["error"].min(),
            pred_df["error"].max(),
            pred_df["error"].std(),
        ],
    })


def error_by_battery(pred_df):
    return (
        pred_df.groupby("battery_id")
        .agg(
            observations=("actual_soh", "size"),
            mean_actual_soh=("actual_soh", "mean"),
            mean_predicted_soh=("predicted_soh", "mean"),
            mean_error=("error", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
            max_absolute_error=("absolute_error", "max"),
        )
        .reset_index()
    )


def error_by_soh_range(pred_df):
    out = (
        pred_df.groupby("soh_range")
        .agg(
            observations=("actual_soh", "size"),
            mean_actual_soh=("actual_soh", "mean"),
            mean_error=("error", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
            max_absolute_error=("absolute_error", "max"),
        )
        .reset_index()
    )
    out["soh_range"] = pd.Categorical(
        out["soh_range"],
        categories=["High_SoH", "Medium_SoH", "Low_SoH"],
        ordered=True,
    )
    return out.sort_values("soh_range").reset_index(drop=True)


def accuracy_thresholds(pred_df, thresholds=None):
    thresholds = thresholds or config.ACCURACY_THRESHOLDS
    rows = []
    for t in thresholds:
        count = int((pred_df["absolute_error"] <= t).sum())
        rows.append({
            "threshold": f"±{t:.2f} SoH",
            "number_predictions": count,
            "percentage": count / len(pred_df) * 100,
        })
    return pd.DataFrame(rows)


def largest_errors(pred_df, n=config.N_LARGEST_ERRORS):
    return (pred_df.sort_values("absolute_error", ascending=False)
            .head(n).copy())


# Plots
def plot_residuals_vs_soh(pred_df, path):
    plt.figure(figsize=(9, 6))
    plt.scatter(pred_df["actual_soh"], pred_df["error"], alpha=0.55)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Actual SoH")
    plt.ylabel("Prediction Error")
    plt.title("Random Forest LOBO Residuals vs Actual SoH")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    save_figure(path)


def plot_absolute_error_vs_soh(pred_df, path):
    plt.figure(figsize=(9, 6))
    plt.scatter(pred_df["actual_soh"], pred_df["absolute_error"], alpha=0.55)
    plt.xlabel("Actual SoH")
    plt.ylabel("Absolute Prediction Error")
    plt.title("Random Forest LOBO Absolute Error vs Actual SoH")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    save_figure(path)


def plot_mae_by_battery(battery_error_df, path):
    plot_df = battery_error_df.sort_values("MAE", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.bar(plot_df["battery_id"], plot_df["MAE"])
    plt.xlabel("Test Battery")
    plt.ylabel("MAE")
    plt.title("Random Forest LOBO MAE by Battery")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    save_figure(path)
