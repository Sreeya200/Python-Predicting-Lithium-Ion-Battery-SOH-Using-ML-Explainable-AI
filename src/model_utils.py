import json
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor
from src import config


# Data loading
def load_modelling_data():
    for file in (config.FEATURE_DATASET_FILE, config.REDUCED_FEATURE_FILE):
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                "Run scripts/00_build_dataset.py first."
            )

    df = pd.read_csv(config.FEATURE_DATASET_FILE)
    selected = pd.read_csv(config.REDUCED_FEATURE_FILE)["feature"].tolist()
    selected = [f for f in selected if f in df.columns]
    return df, selected


# Metrics
def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


# Model factories
def make_rf(n_estimators=300, max_depth=None, min_samples_leaf=2):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=config.RF_MAX_FEATURES,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgb(params):
    return XGBRegressor(**params, random_state=config.RANDOM_STATE, n_jobs=-1)


def rf_lobo_configs(selected_depth):
    return {
        "Baseline Random Forest": dict(config.RF_BASELINE_PARAMS),
        config.TUNED_RF_NAME: {
            "n_estimators": config.RF_TUNED_N_ESTIMATORS,
            "max_depth": selected_depth,
            "min_samples_leaf": config.RF_TUNED_MIN_SAMPLES_LEAF,
        },
    }


# JSON helpers (model configuration files)
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# Plot helper
def save_figure(path):
    plt.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()


# Grouped cross-validation (used for the depth sanity check)
def grouped_cv_evaluate(df, features, make_model, n_splits=config.CV_SPLITS):
    """GroupKFold by battery. ``make_model`` is a zero-argument factory."""
    X = df[features]
    y = df[config.TARGET_COLUMN]
    groups = df[config.GROUP_COLUMN]
    cv = GroupKFold(n_splits=n_splits)

    results = []
    for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y, groups=groups), start=1):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        train_m = calculate_metrics(y.iloc[train_idx],
                                    model.predict(X.iloc[train_idx]))
        valid_m = calculate_metrics(y.iloc[valid_idx],
                                    model.predict(X.iloc[valid_idx]))
        results.append({
            "fold": fold,
            "train_RMSE": train_m["RMSE"],
            "validation_RMSE": valid_m["RMSE"],
            "validation_MAE": valid_m["MAE"],
            "validation_R2": valid_m["R2"],
        })
    return pd.DataFrame(results)


# Holdout evaluation
def split_holdout(df, features):
    """Conventional random 80/20 split (same seed for every model)."""
    return train_test_split(
        df[features], df[config.TARGET_COLUMN],
        test_size=config.HOLDOUT_TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )


# Leave-One-Battery-Out (LOBO)
def run_lobo(df, features, model_factories, verbose=True):
    batteries = sorted(df[config.GROUP_COLUMN].unique())
    target = config.TARGET_COLUMN
    metrics_rows, prediction_frames = [], []
    for model_name, factory in model_factories.items():
        if verbose:
            print(f"\n{model_name}\n{'-' * len(model_name)}")
        for test_battery in batteries:
            train_df = df[df[config.GROUP_COLUMN] != test_battery]
            test_df = df[df[config.GROUP_COLUMN] == test_battery]

            model = factory()
            model.fit(train_df[features], train_df[target])

            train_pred = model.predict(train_df[features])
            test_pred = model.predict(test_df[features])

            test_m = calculate_metrics(test_df[target], test_pred)
            train_rmse = float(np.sqrt(
                mean_squared_error(train_df[target], train_pred)
            ))

            metrics_rows.append({
                "model": model_name,
                "test_battery": test_battery,
                "train_batteries": ", ".join(b for b in batteries
                                             if b != test_battery),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_RMSE": train_rmse,
                "test_RMSE": test_m["RMSE"],
                "test_MAE": test_m["MAE"],
                "test_R2": test_m["R2"],
                "test_SoH_min": test_df[target].min(),
                "test_SoH_max": test_df[target].max(),
            })

            frame = pd.DataFrame({
                "model": model_name,
                "battery_id": test_battery,
                "cycle_number": test_df["cycle_number"].to_numpy(),
                "discharge_sequence": test_df["discharge_sequence"].to_numpy(),
                "actual_soh": test_df[target].to_numpy(),
                "predicted_soh": test_pred,
            })
            frame["prediction_error"] = frame["predicted_soh"] - frame["actual_soh"]
            frame["absolute_error"] = frame["prediction_error"].abs()
            prediction_frames.append(frame)

            if verbose:
                print(f"{test_battery}: RMSE={test_m['RMSE']:.6f}, "
                      f"MAE={test_m['MAE']:.6f}, R²={test_m['R2']:.6f}")

    return (pd.DataFrame(metrics_rows),
            pd.concat(prediction_frames, ignore_index=True))


def pooled_lobo_results(lobo_predictions_df, family, name_map):
    rows = []
    for model_name, evaluation in name_map.items():
        subset = lobo_predictions_df[lobo_predictions_df["model"] == model_name]
        m = calculate_metrics(subset["actual_soh"], subset["predicted_soh"])
        rows.append({"Model": family, "Evaluation": evaluation, **m})
    return pd.DataFrame(rows)


def four_way_comparison(holdout_df, overall_lobo_df):
    combined = pd.concat([holdout_df, overall_lobo_df], ignore_index=True)
    combined["Evaluation"] = pd.Categorical(
        combined["Evaluation"],
        categories=config.EVALUATION_ORDER,
        ordered=True,
    )
    return combined.sort_values("Evaluation").reset_index(drop=True)


# Shared diagnostic plots (RF and XGBoost)
def plot_four_way_rmse(four_way_df, title, path):
    plt.figure(figsize=(10, 6))
    plt.bar(four_way_df["Evaluation"].astype(str), four_way_df["RMSE"])
    plt.xlabel("Evaluation method")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    save_figure(path)


def plot_actual_vs_predicted(tuned_predictions, title, path):
    plt.figure(figsize=(9, 7))
    plt.scatter(tuned_predictions["actual_soh"],
                tuned_predictions["predicted_soh"], alpha=0.65)
    lo = min(tuned_predictions["actual_soh"].min(),
             tuned_predictions["predicted_soh"].min())
    hi = max(tuned_predictions["actual_soh"].max(),
             tuned_predictions["predicted_soh"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual SoH")
    plt.ylabel("Predicted SoH")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    save_figure(path)


def plot_lobo_by_battery(tuned_predictions, batteries, title_prefix,
                         out_dir, file_prefix):
    for battery in batteries:
        res = (tuned_predictions[tuned_predictions["battery_id"] == battery]
               .sort_values("discharge_sequence"))

        plt.figure(figsize=(10, 6))
        plt.plot(res["discharge_sequence"], res["actual_soh"],
                 label="Actual SoH")
        plt.plot(res["discharge_sequence"], res["predicted_soh"],
                 linestyle="--", label="Predicted SoH")
        plt.xlabel("Discharge sequence")
        plt.ylabel("Relative SoH")
        plt.title(f"{title_prefix} — {battery}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        save_figure(out_dir / f"{file_prefix}_{battery}.png")


def plot_error_distribution(tuned_predictions, title, path):
    plt.figure(figsize=(9, 6))
    plt.hist(tuned_predictions["prediction_error"], bins=30)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error (Predicted SoH − Actual SoH)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    save_figure(path)


def plot_feature_importance(importance_df, value_column, xlabel, title, path,
                            top_n=15, figsize=(10, 7)):
    top = importance_df.head(top_n).sort_values(value_column)
    plt.figure(figsize=figsize)
    plt.barh(top["feature"], top[value_column])
    plt.xlabel(xlabel)
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    save_figure(path)
