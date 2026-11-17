"""SHAP explainability for the final Random Forest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src import config
from src import model_utils as mu
TARGET = config.TARGET_COLUMN
GROUP = config.GROUP_COLUMN


def main():
    config.ensure_directories()
    for file in (config.FEATURE_DATASET_FILE, config.RF_MODEL_FILE,
                 config.RF_CONFIG_FILE):
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                "Run scripts/02_RandomForest.py first."
            )

    df = pd.read_csv(config.FEATURE_DATASET_FILE)
    rf_config = mu.load_json(config.RF_CONFIG_FILE)
    features = [f for f in rf_config["features"] if f in df.columns]
    X = df[features].copy()
    y = df[TARGET].copy()
    print(f"Dataset rows   : {len(df)}")
    print(f"Final features : {len(features)}")
    print("\nFeatures used by final RF:")
    for f in features:
        print(f"- {f}")

    print("\nFinal Random Forest configuration:")
    for key in ("n_estimators", "max_depth", "min_samples_leaf",
                "max_features", "random_state"):
        print(f"{key:20s}: {rf_config[key]}")

    final_rf = joblib.load(config.RF_MODEL_FILE)


    # Full-data fit metrics
    full_predictions = final_rf.predict(X)
    full = mu.calculate_metrics(y, full_predictions)
    print("\nFull-data fit metrics:")
    print(f"MAE  : {full['MAE']:.6f}")
    print(f"RMSE : {full['RMSE']:.6f}")
    print(f"R²   : {full['R2']:.6f}")
    pd.DataFrame({
        "metric": ["MAE", "RMSE", "R2"],
        "value": [full["MAE"], full["RMSE"], full["R2"]],
    }).to_csv(config.EVAL_DIR / "final_random_forest_training_metrics.csv",
              index=False)


    # Random Forest feature importance
    importance_df = pd.DataFrame({
        "feature": features,
        "random_forest_importance": final_rf.feature_importances_,
    }).sort_values("random_forest_importance", ascending=False
                   ).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    print("\nRandom Forest feature importance:")
    print(importance_df[["rank", "feature", "random_forest_importance"]]
          .to_string(index=False))

    importance_df.to_csv(
        config.SHAP_DIR / "final_random_forest_feature_importance.csv",
        index=False)

    plot_imp = importance_df.sort_values("random_forest_importance")
    plt.figure(figsize=(9, 10))
    plt.barh(plot_imp["feature"], plot_imp["random_forest_importance"])
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel("Feature")
    plt.title("Final Random Forest Feature Importance")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    mu.save_figure(config.SHAP_DIR / "final_random_forest_feature_importance.png")


    # SHAP values

    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(final_rf)
    shap_values = np.asarray(explainer.shap_values(X))
    expected_value = float(np.ravel(explainer.expected_value)[0])
    print("SHAP calculation complete.")
    print(f"SHAP matrix shape: {shap_values.shape}")
    shap_importance_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_importance_df["rank"] = np.arange(1, len(shap_importance_df) + 1)

    print("\nGlobal SHAP feature importance:")
    print(shap_importance_df[["rank", "feature", "mean_abs_shap"]]
          .to_string(index=False))
    shap_importance_df.to_csv(
        config.SHAP_DIR / "final_random_forest_shap_importance.csv",
        index=False)

    # Summary bar plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Global SHAP Importance — Random Forest")
    plt.tight_layout()
    mu.save_figure(config.SHAP_DIR / "final_random_forest_shap_bar.png")

    # Beeswarm plot
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary — Random Forest SoH Prediction")
    plt.tight_layout()
    mu.save_figure(config.SHAP_DIR / "final_random_forest_shap_beeswarm.png")
    # Dependence plots for the top features
    top_features = (shap_importance_df.head(config.TOP_N_SHAP_DEPENDENCE)
                    ["feature"].tolist())
    print("\nTop SHAP features:")
    for feature in top_features:
        print(f"- {feature}")

    for feature in top_features:
        shap.dependence_plot(feature, shap_values, X,
                             interaction_index=None, show=False)
        plt.title(f"SHAP Dependence — {feature}")
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace(" ", "_")
        mu.save_figure(config.SHAP_DEPENDENCE_DIR
                       / f"shap_dependence_{safe_name}.png")


    # Local explanations: low / medium / high SoH observations
    soh_sorted = y.sort_values()
    representative_cases = [
        (soh_sorted.index[int(len(soh_sorted) * 0.10)], "Low_SoH"),
        (soh_sorted.index[int(len(soh_sorted) * 0.50)], "Medium_SoH"),
        (soh_sorted.index[int(len(soh_sorted) * 0.90)], "High_SoH"),
    ]

    local_records = []
    for index, label in representative_cases:
        row_position = X.index.get_loc(index)
        actual_value = y.loc[index]
        predicted_value = full_predictions[row_position]

        print(f"\n{label}")
        print(f"Battery       : {df.loc[index, GROUP]}")
        print(f"Cycle         : {df.loc[index, 'cycle_number']}")
        print(f"Actual SoH    : {actual_value:.6f}")
        print(f"Predicted SoH : {predicted_value:.6f}")

        row_shap = shap_values[row_position]

        local_df = pd.DataFrame({
            "feature": features,
            "feature_value": X.loc[index].values,
            "shap_value": row_shap,
        })
        local_df["abs_shap"] = np.abs(local_df["shap_value"])
        local_df = local_df.sort_values("abs_shap", ascending=False)
        local_df["case"] = label
        local_df["battery_id"] = df.loc[index, GROUP]
        local_df["cycle_number"] = df.loc[index, "cycle_number"]
        local_df["actual_soh"] = actual_value
        local_df["predicted_soh"] = predicted_value
        local_records.append(local_df)

        explanation = shap.Explanation(
            values=row_shap,
            base_values=expected_value,
            data=X.loc[index].values,
            feature_names=features,
        )
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title(f"SHAP Local Explanation — {label}")
        plt.tight_layout()
        mu.save_figure(config.SHAP_WATERFALL_DIR / f"shap_waterfall_{label}.png")

    pd.concat(local_records, ignore_index=True).to_csv(
        config.SHAP_DIR / "final_random_forest_local_shap_explanations.csv",
        index=False)


    # Complete SHAP matrix
    shap_matrix_df = pd.DataFrame(shap_values, columns=features, index=df.index)
    shap_matrix_df.insert(0, "battery_id", df[GROUP].values)
    shap_matrix_df.insert(1, "cycle_number", df["cycle_number"].values)
    shap_matrix_df.insert(2, "actual_soh", y.values)
    shap_matrix_df.insert(3, "predicted_soh", full_predictions)
    shap_matrix_df.to_csv(
        config.SHAP_DIR / "final_random_forest_shap_values.csv", index=False)
    print(f"\nSHAP outputs saved to: {config.SHAP_DIR}")


if __name__ == "__main__":
    main()
