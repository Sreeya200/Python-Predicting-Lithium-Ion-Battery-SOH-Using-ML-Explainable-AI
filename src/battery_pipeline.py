import json
import pandas as pd
from src import config
from src import data_processing as dp
from src import feature_engineering as fe


def run_dataset_pipeline(verbose=True):
    config.ensure_directories()
    log = print if verbose else (lambda *a, **k: None)

    # Load raw data
    battery_objects = dp.load_battery_objects()
    log("NASA battery files loaded successfully.")
    for battery_id in battery_objects:
        log(f"{battery_id}: loaded")

    # Stage 1: raw inspection
    battery_summary, discharge_inventory, matlab_structure = (
        dp.inspect_batteries(battery_objects)
    )

    battery_summary.to_csv(config.BATTERY_SUMMARY_FILE, index=False)
    discharge_inventory.to_csv(config.DISCHARGE_INVENTORY_FILE, index=False)

    with open(config.EDA_DIR / "matlab_structure.json", "w") as f:
        json.dump(matlab_structure, f, indent=2)

    log("\nStage 1 completed.")
    log(battery_summary.to_string(index=False))

    # Stage 2: cycle-level SoH dataset
    cycle_data = dp.build_cycle_dataset(battery_objects)
    log("\nCycle-level dataset checks:")
    log(f"Total discharge records: {len(cycle_data)}")
    log(f"Number of batteries: {cycle_data['battery_id'].nunique()}")
    log("Duplicate battery/cycle combinations: "
        f"{cycle_data.duplicated(['battery_id', 'cycle_number']).sum()}")
    log(f"Missing capacity: {cycle_data['capacity_ah'].isna().sum()}")
    log(f"Missing SoH: {cycle_data['soh_relative'].isna().sum()}")
    log("\nRecords per battery:")
    log(cycle_data.groupby("battery_id").size().rename("records").to_string())
    log("\nSoH range by battery:")
    log(cycle_data.groupby("battery_id")["soh_relative"]
        .agg(["min", "max"]).to_string())

    cycle_data.to_csv(config.CYCLE_DATASET_FILE, index=False)
    log("\nStage 2 completed: battery_soh_cycle_dataset.csv")

    # Stage 3: feature engineering
    feature_data = fe.build_feature_dataset(battery_objects, cycle_data)
    log("\nFeature dataset shape:")
    log(str(feature_data.shape))
    log("\nMissing values in 24 features:")
    log(feature_data[fe.CANDIDATE_FEATURES].isna().sum().to_string())
    fe.feature_statistics(feature_data).to_csv(config.FEATURE_STATISTICS_FILE)
    fe.feature_dictionary().to_csv(config.FEATURE_DICTIONARY_FILE, index=False)
    feature_data.to_csv(config.FEATURE_DATASET_FILE, index=False)

    log("\nStage 3 completed: 24 engineered features created.")

    # Stage 4: correlation analysis and feature reduction
    corr_matrix = fe.compute_correlation_matrix(feature_data)
    corr_matrix.to_csv(config.CORRELATION_MATRIX_FILE)
    soh_corr = fe.feature_soh_correlations(corr_matrix)
    reduced_features, removal_decisions = fe.reduce_features(corr_matrix)
    removed_features = [
        f for f in fe.CANDIDATE_FEATURES if f not in reduced_features
    ]

    log("\n" + "=" * 60)
    log("AUTOMATIC FEATURE REDUCTION SUMMARY")
    log("=" * 60)
    log(f"Original features : {len(fe.CANDIDATE_FEATURES)}")
    log(f"Removed features  : {len(removed_features)}")
    log(f"Final features    : {len(reduced_features)}")
    log("\nRemoved features:")
    for f in removed_features:
        log(f"- {f}")
    log("\nRetained features:")
    for f in reduced_features:
        log(f"- {f}")

    remaining_pairs = fe.remaining_high_correlation_pairs(
        corr_matrix, reduced_features
    )
    if len(remaining_pairs):
        log("\nWARNING: Highly correlated pairs still remain:")
        log(remaining_pairs.to_string(index=False))
    else:
        log("\nVerification successful: "
            f"no feature pairs with |r| >= {config.CORRELATION_THRESHOLD:.2f} remain.")

    fe.feature_review_table(soh_corr, reduced_features).to_csv(
        config.FEATURE_REVIEW_FILE, index=False
    )
    pd.DataFrame({"feature": reduced_features}).to_csv(
        config.REDUCED_FEATURE_FILE, index=False
    )
    removal_decisions.to_csv(config.FEATURE_REMOVAL_FILE, index=False)

    (soh_corr.sort_values(key=lambda x: abs(x), ascending=False)
     .rename("correlation_with_soh")
     .reset_index()
     .rename(columns={"index": "feature"})
     .to_csv(config.FEATURE_SOH_CORRELATIONS_FILE, index=False))

    log("\n" + "=" * 60)
    log("DATASET PIPELINE FINISHED")
    log("=" * 60)
    log(f"NASA batteries processed    : {len(config.BATTERY_FILES)}")
    log(f"Original engineered features: {len(fe.CANDIDATE_FEATURES)}")
    log(f"Reduced features            : {len(reduced_features)}")

    return {
        "battery_summary": battery_summary,
        "cycle_data": cycle_data,
        "feature_data": feature_data,
        "correlation_matrix": corr_matrix,
        "reduced_features": reduced_features,
    }
