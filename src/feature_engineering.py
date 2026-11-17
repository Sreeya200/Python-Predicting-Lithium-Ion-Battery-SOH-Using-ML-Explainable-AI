import numpy as np
import pandas as pd
from src.config import CORRELATION_THRESHOLD, TARGET_COLUMN
from src.data_processing import (
    extract_cycle_data,
    get_cycle_type,
    get_cycles,
)

# Feature definitions
CANDIDATE_FEATURES = [
    "dis_voltage_q10",
    "dis_voltage_q50",
    "dis_voltage_q90",
    "dis_voltage_iqr",
    "dis_voltage_range",
    "dis_voltage_std",
    "dis_voltage_slope",
    "dis_current_mean",
    "dis_current_std",
    "dis_current_range",
    "dis_temp_mean",
    "dis_temp_max",
    "dis_temp_rise",
    "dis_temp_slope",
    "dis_duration",
    "dis_voltage_time_area",
    "chg_voltage_mean",
    "chg_voltage_std",
    "chg_current_mean",
    "chg_current_std",
    "chg_temp_mean",
    "chg_temp_max",
    "chg_duration",
    "chg_voltage_time_area",
]

FEATURE_DESCRIPTIONS = [
    "10th percentile of discharge voltage",
    "Median of discharge voltage",
    "90th percentile of discharge voltage",
    "Interquartile range of discharge voltage",
    "Range of discharge voltage",
    "Standard deviation of discharge voltage",
    "Slope of discharge voltage over time",
    "Mean discharge current",
    "Standard deviation of discharge current",
    "Range of discharge current",
    "Mean discharge temperature",
    "Maximum discharge temperature",
    "Discharge temperature rise",
    "Slope of discharge temperature over time",
    "Discharge duration",
    "Area under discharge voltage-time curve",
    "Mean voltage during preceding charge",
    "Standard deviation of voltage during preceding charge",
    "Mean current during preceding charge",
    "Standard deviation of current during preceding charge",
    "Mean temperature during preceding charge",
    "Maximum temperature during preceding charge",
    "Preceding charge duration",
    "Area under preceding charge voltage-time curve",
]


# Safe statistics helpers
def safe_mean(x):
    return np.mean(x) if len(x) else np.nan
def safe_std(x):
    return np.std(x, ddof=1) if len(x) > 1 else np.nan
def safe_min(x):
    return np.min(x) if len(x) else np.nan
def safe_max(x):
    return np.max(x) if len(x) else np.nan
def safe_range(x):
    return np.max(x) - np.min(x) if len(x) else np.nan
def safe_percentile(x, q):
    return np.percentile(x, q) if len(x) else np.nan


def safe_slope(x, t):
    if len(x) < 2 or len(t) != len(x):
        return np.nan
    try:
        return np.polyfit(t, x, 1)[0]
    except Exception:
        return np.nan


def safe_trapz(y, x):
    if len(y) < 2 or len(x) != len(y):
        return np.nan
    try:
        return np.trapezoid(y, x)
    except AttributeError:          # NumPy < 2.0
        return np.trapz(y, x)


# Feature extraction
def _discharge_features(discharge):
    voltage = discharge["voltage"]
    current = discharge["current"]
    temperature = discharge["temperature"]
    time = discharge["time"]
    q10 = safe_percentile(voltage, 10)
    q50 = safe_percentile(voltage, 50)
    q90 = safe_percentile(voltage, 90)
    iqr = q90 - q10 if pd.notna(q90) and pd.notna(q10) else np.nan

    return {
        "dis_voltage_q10": q10,
        "dis_voltage_q50": q50,
        "dis_voltage_q90": q90,
        "dis_voltage_iqr": iqr,
        "dis_voltage_range": safe_range(voltage),
        "dis_voltage_std": safe_std(voltage),
        "dis_voltage_slope": safe_slope(voltage, time),
        "dis_current_mean": safe_mean(current),
        "dis_current_std": safe_std(current),
        "dis_current_range": safe_range(current),
        "dis_temp_mean": safe_mean(temperature),
        "dis_temp_max": safe_max(temperature),
        "dis_temp_rise":
            temperature[-1] - temperature[0] if len(temperature) >= 2 else np.nan,
        "dis_temp_slope": safe_slope(temperature, time),
        "dis_duration": time[-1] - time[0] if len(time) >= 2 else np.nan,
        "dis_voltage_time_area": safe_trapz(voltage, time),
    }


def _charge_features(charge):
    if charge is None:
        return {
            "chg_voltage_mean": np.nan, "chg_voltage_std": np.nan,
            "chg_current_mean": np.nan, "chg_current_std": np.nan,
            "chg_temp_mean": np.nan, "chg_temp_max": np.nan,
            "chg_duration": np.nan, "chg_voltage_time_area": np.nan,
        }

    voltage = charge["voltage"]
    current = charge["current"]
    temperature = charge["temperature"]
    time = charge["time"]
    return {
        "chg_voltage_mean": safe_mean(voltage),
        "chg_voltage_std": safe_std(voltage),
        "chg_current_mean": safe_mean(current),
        "chg_current_std": safe_std(current),
        "chg_temp_mean": safe_mean(temperature),
        "chg_temp_max": safe_max(temperature),
        "chg_duration": time[-1] - time[0] if len(time) >= 2 else np.nan,
        "chg_voltage_time_area": safe_trapz(voltage, time),
    }


def build_feature_dataset(battery_objects, cycle_data):
    feature_records = []
    for battery_id, battery in battery_objects.items():
        cycles = get_cycles(battery)
        latest_charge = None
        for cycle_number, cycle in enumerate(cycles, start=1):
            cycle_type = get_cycle_type(cycle)
            if cycle_type == "charge":
                latest_charge = extract_cycle_data(cycle)
                continue
            if cycle_type != "discharge":
                continue
            discharge = extract_cycle_data(cycle)
            valid_capacity = discharge["capacity"][
                np.isfinite(discharge["capacity"])
            ]
            if len(valid_capacity) == 0:
                continue
            record = {
                "battery_id": battery_id,
                "cycle_number": cycle_number,
                "capacity_ah": valid_capacity[-1],
            }
            record.update(_discharge_features(discharge))
            record.update(_charge_features(latest_charge))
            feature_records.append(record)
    features = pd.DataFrame(feature_records)

    soh_reference = cycle_data[[
        "battery_id", "cycle_number", "soh_relative",
        "initial_capacity_ah", "discharge_sequence",
    ]].copy()

    return features.merge(
        soh_reference,
        on=["battery_id", "cycle_number"],
        how="left",
        validate="one_to_one",
    )


def feature_statistics(feature_data, features=None):
    features = features or CANDIDATE_FEATURES
    stats = feature_data[features].describe().T
    stats["missing"] = feature_data[features].isna().sum()
    return stats


def feature_dictionary():
    return pd.DataFrame({
        "feature": CANDIDATE_FEATURES,
        "description": FEATURE_DESCRIPTIONS,
    })


# Correlation analysis and automatic feature reduction
def compute_correlation_matrix(feature_data, features=None):
    features = features or CANDIDATE_FEATURES
    return feature_data[features + [TARGET_COLUMN]].corr(method="pearson")
def feature_soh_correlations(correlation_matrix):
    return correlation_matrix[TARGET_COLUMN].drop(TARGET_COLUMN)


def reduce_features(correlation_matrix, features=None,
                    threshold=CORRELATION_THRESHOLD):
    features = features or CANDIDATE_FEATURES
    soh_corr = feature_soh_correlations(correlation_matrix)
    feature_corr = correlation_matrix.loc[features, features]
    remaining = list(features)
    decisions = []
    while True:
        current = feature_corr.loc[remaining, remaining]
        upper = current.where(
            np.triu(np.ones(current.shape), k=1).astype(bool)
        )
        strongest = upper.stack().abs().sort_values(ascending=False)
        if strongest.empty:
            break
        feature_1, feature_2 = strongest.index[0]
        actual_corr = current.loc[feature_1, feature_2]
        if abs(actual_corr) < threshold:
            break
        corr_1 = abs(soh_corr[feature_1])
        corr_2 = abs(soh_corr[feature_2])
        if corr_1 >= corr_2:
            retained, removed = feature_1, feature_2
        else:
            retained, removed = feature_2, feature_1
        decisions.append({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "pairwise_correlation": actual_corr,
            "feature_1_abs_soh_correlation": corr_1,
            "feature_2_abs_soh_correlation": corr_2,
            "retained_feature": retained,
            "removed_feature": removed,
        })

        remaining.remove(removed)

    return remaining, pd.DataFrame(decisions)


def remaining_high_correlation_pairs(correlation_matrix, reduced_features,
                                     threshold=CORRELATION_THRESHOLD):
    sub = correlation_matrix.loc[reduced_features, reduced_features]
    pairs = []
    for i, f1 in enumerate(reduced_features):
        for f2 in reduced_features[i + 1:]:
            r = sub.loc[f1, f2]
            if abs(r) >= threshold:
                pairs.append({"feature_1": f1, "feature_2": f2,
                              "correlation": r})
    return pd.DataFrame(pairs)


def high_correlation_pairs(correlation_matrix, features=None,
                           threshold=CORRELATION_THRESHOLD):
    features = features or CANDIDATE_FEATURES
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = features[i], features[j]
            r = correlation_matrix.loc[a, b]
            if abs(r) >= threshold:
                pairs.append({
                    "feature_1": a, "feature_2": b,
                    "correlation": r, "absolute_correlation": abs(r),
                })

    cols = ["feature_1", "feature_2", "correlation", "absolute_correlation"]
    if not pairs:
        return pd.DataFrame(columns=cols)

    return (pd.DataFrame(pairs)
            .sort_values("absolute_correlation", ascending=False)
            .reset_index(drop=True))


def feature_review_table(soh_corr, reduced_features, features=None):
    features = features or CANDIDATE_FEATURES
    return pd.DataFrame({
        "feature": features,
        "decision": [
            "Retained" if f in reduced_features else "Removed for high correlation"
            for f in features
        ],
        "absolute_correlation_with_soh": [abs(soh_corr[f]) for f in features],
    })
