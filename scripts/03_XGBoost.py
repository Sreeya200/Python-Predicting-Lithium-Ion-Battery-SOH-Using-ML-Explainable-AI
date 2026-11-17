"""XGBoost: baseline vs tuned, holdout, LOBO, feature importance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from src import config
from src import model_utils as mu
TARGET = config.TARGET_COLUMN
GROUP = config.GROUP_COLUMN


def main():
    config.ensure_directories()
    df, features = mu.load_modelling_data()
    X, y = df[features], df[TARGET]
    batteries = sorted(df[GROUP].unique())
    configs = config.XGB_CONFIGS
    print("XGBOOST MODELLING")
    print(f"Dataset rows       : {len(df)}")
    print(f"Reduced features   : {len(features)}")
    print(f"Batteries          : {batteries}")
    print(f"\nSelected features:\n{features}")
    for name, params in configs.items():
        print(f"\n{name}:\n{params}")


    # Conventional 80/20 holdout
    X_train, X_test, y_train, y_test = mu.split_holdout(df, features)
    holdout_results = {}
    for name, params in configs.items():
        model = mu.make_xgb(params)
        model.fit(X_train, y_train)
        holdout_results[name] = mu.calculate_metrics(y_test, model.predict(X_test))

    holdout_df = pd.DataFrame([
        {"Model": "XGBoost", "Evaluation": "Baseline holdout",
         **holdout_results["Baseline XGBoost"]},
        {"Model": "XGBoost", "Evaluation": "Tuned holdout",
         **holdout_results["Tuned XGBoost"]},
    ])
    holdout_df.to_csv(config.XGB_DIR / "xgboost_baseline_tuned_holdout.csv",
                      index=False)

    print("\nXGBOOST BASELINE VS TUNED HOLDOUT")
    print(holdout_df.round(6).to_string(index=False))

    base, tuned = (holdout_results["Baseline XGBoost"],
                   holdout_results["Tuned XGBoost"])
    print("\nHOLDOUT IMPROVEMENT")
    print(f"RMSE improvement: {(base['RMSE'] - tuned['RMSE']) / base['RMSE'] * 100:.2f}%")
    print(f"MAE improvement: {(base['MAE'] - tuned['MAE']) / base['MAE'] * 100:.2f}%")


    # Leave-One-Battery-Out
    factories = {
        name: (lambda p=params: mu.make_xgb(p))
        for name, params in configs.items()
    }
    lobo_metrics_df, lobo_predictions_df = mu.run_lobo(df, features, factories)

    print("\nBATTERY-LEVEL XGBOOST LOBO RESULTS")
    print(lobo_metrics_df.round(6).to_string(index=False))

    overall_lobo_df = mu.pooled_lobo_results(
        lobo_predictions_df, "XGBoost",
        {"Baseline XGBoost": "Baseline LOBO", "Tuned XGBoost": "Tuned LOBO"},
    )
    print("\nOVERALL XGBOOST LOBO RESULTS")
    print(overall_lobo_df.round(6).to_string(index=False))

    lobo_metrics_df.to_csv(
        config.XGB_DIR / "xgboost_baseline_tuned_lobo_metrics.csv", index=False)
    lobo_predictions_df.to_csv(
        config.XGB_DIR / "xgboost_baseline_tuned_lobo_predictions.csv",
        index=False)
    overall_lobo_df.to_csv(
        config.XGB_DIR / "xgboost_baseline_tuned_lobo_overall.csv", index=False)

    b = overall_lobo_df[overall_lobo_df["Evaluation"] == "Baseline LOBO"].iloc[0]
    t = overall_lobo_df[overall_lobo_df["Evaluation"] == "Tuned LOBO"].iloc[0]
    print("\nBASELINE VS TUNED XGBOOST LOBO")
    print(f"Baseline LOBO RMSE: {b['RMSE']:.6f}")
    print(f"Tuned LOBO RMSE: {t['RMSE']:.6f}")
    print(f"RMSE improvement: {(b['RMSE'] - t['RMSE']) / b['RMSE'] * 100:.2f}%")
    print(f"MAE improvement: {(b['MAE'] - t['MAE']) / b['MAE'] * 100:.2f}%")


    # Four-way comparison and diagnostic plots

    four_way = mu.four_way_comparison(holdout_df, overall_lobo_df)
    four_way.to_csv(config.XGB_DIR / "xgboost_four_way_comparison.csv",
                    index=False)
    print("\nXGBOOST — OVERALL COMPARISON")
    print(four_way.round(6).to_string(index=False))
    mu.plot_four_way_rmse(
        four_way, "XGBoost Baseline and Tuned Performance",
        config.XGB_DIR / "xgboost_four_way_rmse_comparison.png")
    tuned_predictions = lobo_predictions_df[
        lobo_predictions_df["model"] == "Tuned XGBoost"]
    mu.plot_actual_vs_predicted(
        tuned_predictions, "Tuned XGBoost Leave-One-Battery-Out Predictions",
        config.XGB_DIR / "xgboost_lobo_actual_vs_predicted.png")
    mu.plot_lobo_by_battery(
        tuned_predictions, batteries, "Tuned XGBoost LOBO SoH Prediction",
        config.XGB_LOBO_DIR, "xgboost_lobo")
    mu.plot_error_distribution(
        tuned_predictions, "Tuned XGBoost LOBO Prediction Error Distribution",
        config.XGB_DIR / "xgboost_lobo_error_distribution.png")


    # Final model (fit on all data): importance + saved artefacts
    tuned_params = configs["Tuned XGBoost"]
    final_xgb = mu.make_xgb(tuned_params)
    final_xgb.fit(X, y)
    importance_df = pd.DataFrame({
        "feature": features,
        "xgboost_importance": final_xgb.feature_importances_,
    }).sort_values("xgboost_importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(len(importance_df)) + 1
    importance_df.to_csv(config.XGB_DIR / "xgboost_feature_importance.csv",
                         index=False)

    mu.plot_feature_importance(
        importance_df, "xgboost_importance", "XGBoost Feature Importance",
        "Top 15 XGBoost Feature Importance",
        config.XGB_DIR / "xgboost_feature_importance.png", figsize=(9, 7))
    configuration = pd.DataFrame({
        "parameter": ["feature_count", *tuned_params.keys(), "random_state"],
        "value": [len(features), *tuned_params.values(), config.RANDOM_STATE],
    })
    configuration.to_csv(config.XGB_DIR / "xgboost_final_configuration.csv",
                         index=False)

    joblib.dump(final_xgb, config.XGB_MODEL_FILE)
    mu.save_json({
        "model": "XGBRegressor",
        "features": features,
        **tuned_params,
        "random_state": config.RANDOM_STATE,
        "target": TARGET,
    }, config.XGB_CONFIG_FILE)

    print("\nTop XGBoost features:")
    print(importance_df.head(15).round(6).to_string(index=False))
    print(f"\nXGBoost outputs saved to: {config.XGB_DIR}")
    print(f"Model saved to: {config.XGB_MODEL_FILE}")


if __name__ == "__main__":
    main()
