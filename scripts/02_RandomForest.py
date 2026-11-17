"""Random Forest: depth check, holdout, LOBO, feature importance.

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
from src import config
from src import model_utils as mu
TARGET = config.TARGET_COLUMN
GROUP = config.GROUP_COLUMN


def depth_sanity_check(df, features):
    rows = []
    for depth in config.RF_DEPTH_VALUES:
        fold_df = mu.grouped_cv_evaluate(
            df, features,
            lambda d=depth: mu.make_rf(
                n_estimators=config.RF_TUNED_N_ESTIMATORS,
                max_depth=d,
                min_samples_leaf=config.RF_TUNED_MIN_SAMPLES_LEAF,
            ),
        )
        rows.append({
            "max_depth": "None" if depth is None else str(depth),
            "mean_train_RMSE": fold_df["train_RMSE"].mean(),
            "mean_validation_RMSE": fold_df["validation_RMSE"].mean(),
            "std_validation_RMSE": fold_df["validation_RMSE"].std(),
            "mean_validation_MAE": fold_df["validation_MAE"].mean(),
            "mean_validation_R2": fold_df["validation_R2"].mean(),
        })

    depth_df = pd.DataFrame(rows)
    depth_df["depth_numeric"] = (
        depth_df["max_depth"].replace("None", np.nan).astype(float)
    )
    depth_df = (depth_df.sort_values("depth_numeric", na_position="last")
                .drop(columns="depth_numeric").reset_index(drop=True))
    depth_df["overfit_gap"] = (
        depth_df["mean_validation_RMSE"] - depth_df["mean_train_RMSE"]
    )
    return depth_df


def plot_depth_check(depth_df):
    finite = depth_df[depth_df["max_depth"] != "None"]
    unrestricted = depth_df[depth_df["max_depth"] == "None"]
    plt.figure(figsize=(10, 6))
    plt.plot(finite["max_depth"].astype(int), finite["mean_validation_RMSE"],
             marker="o", label="Finite max_depth")

    if len(unrestricted) == 1:
        x_none = finite["max_depth"].astype(int).max() + 3
        y_none = unrestricted["mean_validation_RMSE"].iloc[0]
        plt.scatter(x_none, y_none, marker="X", s=100, label="max_depth=None")
        plt.annotate("None", (x_none, y_none), xytext=(8, 8),
                     textcoords="offset points")
    plt.xlabel("Maximum tree depth")
    plt.ylabel("Mean validation RMSE")
    plt.title("Random Forest Validation RMSE by Tree Depth")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    mu.save_figure(config.RF_DIR / "random_forest_depth_sanity_check.png")


def main():
    config.ensure_directories()
    df, features = mu.load_modelling_data()
    y = df[TARGET]
    batteries = sorted(df[GROUP].unique())

    print("RANDOM FOREST MODELLING")
    print(f"Dataset rows     : {len(df)}")
    print(f"Reduced features : {len(features)}")
    print(f"Features         : {features}")
    print(f"Batteries        : {batteries}")

    # Depth sanity check (GroupKFold by battery)
    depth_df = depth_sanity_check(df, features)

    best_row = depth_df.loc[depth_df["mean_validation_RMSE"].idxmin()]
    selected_depth = (None if best_row["max_depth"] == "None"
                      else int(best_row["max_depth"]))

    print("\nDEPTH SANITY CHECK")
    print(depth_df.round(6).to_string(index=False))
    print(f"\nSelected max_depth: {selected_depth}")

    depth_df.to_csv(config.RF_DIR / "random_forest_depth_sanity_check.csv",
                    index=False)
    plot_depth_check(depth_df)

    # Baseline vs tuned 80/20 holdout
    X_train, X_test, y_train, y_test = mu.split_holdout(df, features)
    configs = mu.rf_lobo_configs(selected_depth)
    holdout_rows = []
    for evaluation, model_name in [("Baseline holdout", "Baseline Random Forest"),
                                   ("Tuned holdout", config.TUNED_RF_NAME)]:
        model = mu.make_rf(**configs[model_name])
        model.fit(X_train, y_train)
        m = mu.calculate_metrics(y_test, model.predict(X_test))
        holdout_rows.append({"Model": "Random Forest",
                             "Evaluation": evaluation, **m})

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(config.RF_DIR / "random_forest_baseline_tuned_holdout.csv",
                      index=False)

    print("\nRANDOM FOREST BASELINE VS TUNED HOLDOUT")
    print(holdout_df.round(6).to_string(index=False))

    base_rmse = holdout_df.loc[holdout_df["Evaluation"] == "Baseline holdout",
                               "RMSE"].iloc[0]
    tuned_rmse = holdout_df.loc[holdout_df["Evaluation"] == "Tuned holdout",
                                "RMSE"].iloc[0]
    print(f"\nHoldout RMSE improvement: "
          f"{(base_rmse - tuned_rmse) / base_rmse * 100:.2f}%")

    # Final model (fit on all data): importance + saved artefacts
    final_rf = mu.make_rf(
        n_estimators=config.RF_TUNED_N_ESTIMATORS,
        max_depth=selected_depth,
        min_samples_leaf=config.RF_TUNED_MIN_SAMPLES_LEAF,
    )
    final_rf.fit(df[features], y)

    importance_df = pd.DataFrame({
        "feature": features,
        "random_forest_importance": final_rf.feature_importances_,
    }).sort_values("random_forest_importance", ascending=False
                   ).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    importance_df.to_csv(config.RF_DIR / "random_forest_feature_importance.csv",
                         index=False)

    print("\nRANDOM FOREST FEATURE IMPORTANCE")
    print(importance_df.round(6).to_string(index=False))
    mu.plot_feature_importance(
        importance_df, "random_forest_importance",
        "Random Forest feature importance", "Random Forest Feature Importance",
        config.RF_DIR / "random_forest_feature_importance.png",
    )

    configuration = pd.DataFrame({
        "parameter": ["feature_source", "feature_count", "n_estimators",
                      "max_depth", "min_samples_leaf", "max_features",
                      "random_state"],
        "value": ["automatic_reduced_feature_set.csv", len(features),
                  config.RF_TUNED_N_ESTIMATORS, str(selected_depth),
                  config.RF_TUNED_MIN_SAMPLES_LEAF, config.RF_MAX_FEATURES,
                  config.RANDOM_STATE],
    })
    configuration.to_csv(config.RF_DIR / "random_forest_final_configuration.csv",
                         index=False)

    joblib.dump(final_rf, config.RF_MODEL_FILE)
    mu.save_json({
        "model": "RandomForestRegressor",
        "features": features,
        "n_estimators": config.RF_TUNED_N_ESTIMATORS,
        "max_depth": selected_depth,
        "min_samples_leaf": config.RF_TUNED_MIN_SAMPLES_LEAF,
        "max_features": config.RF_MAX_FEATURES,
        "random_state": config.RANDOM_STATE,
        "target": TARGET,
    }, config.RF_CONFIG_FILE)


    # Leave-One-Battery-Out
    factories = {
        name: (lambda p=params: mu.make_rf(**p))
        for name, params in configs.items()
    }
    lobo_metrics_df, lobo_predictions_df = mu.run_lobo(
        df, features, factories, verbose=False
    )

    print("\nBATTERY-LEVEL LOBO RESULTS")
    print(lobo_metrics_df.round(6).to_string(index=False))

    overall_lobo_df = mu.pooled_lobo_results(
        lobo_predictions_df, "Random Forest",
        {"Baseline Random Forest": "Baseline LOBO",
         config.TUNED_RF_NAME: "Tuned LOBO"},
    )
    print("\nOVERALL LOBO RESULTS")
    print(overall_lobo_df.round(6).to_string(index=False))

    lobo_metrics_df.to_csv(
        config.RF_DIR / "random_forest_baseline_tuned_lobo_metrics.csv",
        index=False)
    lobo_predictions_df.to_csv(
        config.RF_DIR / "random_forest_baseline_tuned_lobo_predictions.csv",
        index=False)
    overall_lobo_df.to_csv(
        config.RF_DIR / "random_forest_baseline_tuned_lobo_overall.csv",
        index=False)

    # Four-way comparison and diagnostic plots
    four_way = mu.four_way_comparison(holdout_df, overall_lobo_df)
    four_way.to_csv(config.RF_DIR / "random_forest_four_way_comparison.csv",
                    index=False)
    print("\nRANDOM FOREST — OVERALL COMPARISON")
    print(four_way.round(6).to_string(index=False))
    mu.plot_four_way_rmse(
        four_way, "Random Forest Baseline and Tuned Performance",
        config.RF_DIR / "random_forest_four_way_rmse_comparison.png")
    tuned_predictions = lobo_predictions_df[
        lobo_predictions_df["model"] == config.TUNED_RF_NAME]
    mu.plot_actual_vs_predicted(
        tuned_predictions,
        "Tuned Random Forest Leave-One-Battery-Out Predictions",
        config.RF_DIR / "random_forest_lobo_actual_vs_predicted.png")

    mu.plot_lobo_by_battery(
        tuned_predictions, batteries,
        "Tuned Random Forest LOBO SoH Prediction",
        config.RF_LOBO_DIR, "random_forest_lobo")

    mu.plot_error_distribution(
        tuned_predictions,
        "Tuned Random Forest LOBO Prediction Error Distribution",
        config.RF_DIR / "random_forest_lobo_error_distribution.png")

    print(f"\nRandom Forest outputs saved to: {config.RF_DIR}")
    print(f"Model saved to: {config.RF_MODEL_FILE}")


if __name__ == "__main__":
    main()
