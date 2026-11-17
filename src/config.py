from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
EDA_SCATTER_DIR = EDA_DIR / "eda_feature_vs_soh_plots"
RF_DIR = OUTPUTS_DIR / "random_forest"
RF_LOBO_DIR = RF_DIR / "random_forest_lobo_by_battery"
XGB_DIR = OUTPUTS_DIR / "xgboost"
XGB_LOBO_DIR = XGB_DIR / "xgboost_lobo_by_battery"
SHAP_DIR = OUTPUTS_DIR / "shap"
SHAP_DEPENDENCE_DIR = SHAP_DIR / "shap_dependence"
SHAP_WATERFALL_DIR = SHAP_DIR / "shap_waterfall"
EVAL_DIR = OUTPUTS_DIR / "evaluation"

ALL_DIRECTORIES = [
    RAW_DIR, PROCESSED_DIR, MODELS_DIR, EDA_DIR, EDA_SCATTER_DIR,
    RF_DIR, RF_LOBO_DIR, XGB_DIR, XGB_LOBO_DIR,
    SHAP_DIR, SHAP_DEPENDENCE_DIR, SHAP_WATERFALL_DIR, EVAL_DIR,
]


def ensure_directories():
    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


# Raw NASA data
BATTERY_FILES = {
    "B0005": RAW_DIR / "B0005.mat",
    "B0006": RAW_DIR / "B0006.mat",
    "B0007": RAW_DIR / "B0007.mat",
    "B0018": RAW_DIR / "B0018.mat",
}

# Documented discharge cut-off voltage (V) for each NASA battery
DISCHARGE_CUTOFFS = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5,
}

# Processed files (data/processed)

BATTERY_SUMMARY_FILE = PROCESSED_DIR / "battery_summary.csv"
DISCHARGE_INVENTORY_FILE = PROCESSED_DIR / "discharge_cycle_inventory.csv"
CYCLE_DATASET_FILE = PROCESSED_DIR / "battery_soh_cycle_dataset.csv"
FEATURE_DATASET_FILE = PROCESSED_DIR / "battery_soh_feature_dataset.csv"
FEATURE_DICTIONARY_FILE = PROCESSED_DIR / "feature_dictionary.csv"
FEATURE_STATISTICS_FILE = PROCESSED_DIR / "feature_statistics.csv"
CORRELATION_MATRIX_FILE = PROCESSED_DIR / "complete_correlation_matrix.csv"
FEATURE_SOH_CORRELATIONS_FILE = PROCESSED_DIR / "feature_soh_correlations.csv"
REDUCED_FEATURE_FILE = PROCESSED_DIR / "automatic_reduced_feature_set.csv"
FEATURE_REVIEW_FILE = PROCESSED_DIR / "automatic_correlation_feature_review.csv"
FEATURE_REMOVAL_FILE = PROCESSED_DIR / "automatic_feature_removal_decisions.csv"


# Saved models (models/)
RF_MODEL_FILE = MODELS_DIR / "random_forest_soh.joblib"
RF_CONFIG_FILE = MODELS_DIR / "random_forest_config.json"
XGB_MODEL_FILE = MODELS_DIR / "xgboost_soh.joblib"
XGB_CONFIG_FILE = MODELS_DIR / "xgboost_config.json"


# Modelling constants
RANDOM_STATE = 42
TARGET_COLUMN = "soh_relative"
GROUP_COLUMN = "battery_id"
CORRELATION_THRESHOLD = 0.90      
HOLDOUT_TEST_SIZE = 0.20
CV_SPLITS = 4
TOP_N_SCATTER_FEATURES = 5
TOP_N_SHAP_DEPENDENCE = 5

# Random Forest
RF_DEPTH_VALUES = [3, 5, 8, 12, 16, 20, None]
RF_BASELINE_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
RF_TUNED_N_ESTIMATORS = 300
RF_TUNED_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"

# XGBoost
XGB_CONFIGS = {
    "Baseline XGBoost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.10,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    },
    "Tuned XGBoost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    },
}

EVALUATION_ORDER = [
    "Baseline holdout",
    "Tuned holdout",
    "Baseline LOBO",
    "Tuned LOBO",
]

# Evaluation
TUNED_RF_NAME = "Tuned Random Forest"
SOH_HIGH_THRESHOLD = 0.90
SOH_MEDIUM_THRESHOLD = 0.75
ACCURACY_THRESHOLDS = [0.02, 0.05, 0.10]
N_LARGEST_ERRORS = 20
PLOT_DPI = 300
