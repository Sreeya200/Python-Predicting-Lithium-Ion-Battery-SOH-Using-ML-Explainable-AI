""" Exploratory data analysis plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import config
from src.feature_engineering import CANDIDATE_FEATURES, high_correlation_pairs
from src.model_utils import save_figure
TARGET = config.TARGET_COLUMN


def plot_soh_distribution(feature_data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=feature_data, x="battery_id", y=TARGET)
    plt.xlabel("Battery")
    plt.ylabel("Relative SoH")
    plt.title("Distribution of Relative SoH by Battery")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    save_figure(config.EDA_DIR / "eda_soh_distribution_by_battery.png")


def plot_degradation_trajectories(feature_data):
    plt.figure(figsize=(11, 6))
    for battery in sorted(feature_data["battery_id"].unique()):
        battery_df = (feature_data[feature_data["battery_id"] == battery]
                      .sort_values("discharge_sequence"))
        plt.plot(battery_df["discharge_sequence"], battery_df[TARGET],
                 label=battery)
    plt.xlabel("Discharge sequence")
    plt.ylabel("Relative SoH")
    plt.title("Battery SoH Degradation Trajectories")
    plt.legend(title="Battery")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    save_figure(config.EDA_DIR / "eda_soh_degradation_trajectories.png")


def plot_feature_soh_correlations(feature_soh_corr):
    plot_corr = feature_soh_corr.sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(plot_corr.index, plot_corr.values)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Pearson correlation with SoH")
    plt.ylabel("Feature")
    plt.title("Engineered Feature Correlation with Relative SoH")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    save_figure(config.EDA_DIR / "eda_feature_soh_correlations.png")


def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(20, 17))
    sns.heatmap(
        correlation_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", square=True, linewidths=0.3,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Correlation Matrix of Engineered Battery Features and SoH",
              fontsize=16, pad=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    save_figure(config.EDA_DIR / "feature_correlation_heatmap.png")


def plot_top_feature_scatter(feature_data, feature_soh_corr):
    top_features = (feature_soh_corr.abs()
                    .sort_values(ascending=False)
                    .head(config.TOP_N_SCATTER_FEATURES)
                    .index.tolist())

    for feature in top_features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=feature_data, x=feature, y=TARGET, alpha=0.55)
        plt.xlabel(feature)
        plt.ylabel("Relative SoH")
        plt.title(f"{feature} vs Relative SoH")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace(" ", "_")
        save_figure(config.EDA_SCATTER_DIR / f"eda_{safe_name}_vs_soh.png")


def plot_feature_distributions(feature_data):
    plot_data = feature_data[CANDIDATE_FEATURES].copy()
    plot_data = plot_data.apply(
        lambda c: (c - c.mean()) / c.std() if c.std() != 0 else c
    )
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=plot_data, orient="h")
    plt.xlabel("Standardised feature value")
    plt.ylabel("Feature")
    plt.title("Distribution of Engineered Features")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    save_figure(config.EDA_DIR / "eda_engineered_feature_distributions.png")


def plot_high_feature_correlations(correlation_matrix):
    high_pairs = high_correlation_pairs(correlation_matrix)

    if len(high_pairs) == 0:
        return None
    top_pairs = high_pairs.head(10).copy()
    top_pairs["pair"] = top_pairs["feature_1"] + " vs " + top_pairs["feature_2"]
    top_pairs = top_pairs.sort_values("absolute_correlation")
    plt.figure(figsize=(10, 7))
    plt.barh(top_pairs["pair"], top_pairs["absolute_correlation"])
    plt.xlabel("Absolute Pearson correlation")
    plt.ylabel("Feature pair")
    plt.title("Strongest Feature-to-Feature Correlations")
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    save_figure(config.EDA_DIR / "eda_high_feature_correlations.png")


def main():
    config.ensure_directories()
    for file in (config.FEATURE_DATASET_FILE, config.CORRELATION_MATRIX_FILE):
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                "Run scripts/00_build_dataset.py first."
            )

    feature_data = pd.read_csv(config.FEATURE_DATASET_FILE)
    correlation_matrix = pd.read_csv(config.CORRELATION_MATRIX_FILE, index_col=0)
    feature_soh_corr = correlation_matrix[TARGET].drop(TARGET)
    plot_soh_distribution(feature_data)
    plot_degradation_trajectories(feature_data)
    plot_feature_soh_correlations(feature_soh_corr)
    plot_correlation_heatmap(correlation_matrix)
    plot_top_feature_scatter(feature_data, feature_soh_corr)
    plot_feature_distributions(feature_data)
    plot_high_feature_correlations(correlation_matrix)

    print(f"EDA plots saved to: {config.EDA_DIR}")


if __name__ == "__main__":
    main()
