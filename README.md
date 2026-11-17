# EV Battery State-of-Health (SoH) Prediction

Machine-learning pipeline that predicts the **relative State of Health (SoH)** of
lithium-ion batteries from charge/discharge signals in the
(cells **B0005, B0006, B0007, B0018**).
Two models are compared — **Random Forest** and **XGBoost** — using both a
conventional random 80/20 holdout and a stricter **Leave-One-Battery-Out (LOBO)**
validation that tests generalisation to a battery the model has never seen.
**SHAP** is used to explain the final Random Forest.

> Relative SoH = discharge capacity ÷ the first discharge capacity of the same battery.

---

## Project structure

```
ev-battery-soh-prediction/
├── data/
│   ├── raw/                 # NASA .mat files (B0005, B0006, B0007, B0018)
│   └── processed/           # datasets, feature tables, correlation + reduction results
├── models/                  # saved final models (.joblib) and their configs (.json)
├── outputs/
│   ├── eda/                 # exploratory plots
│   ├── random_forest/       # depth check, holdout, LOBO, importance, plots
│   ├── xgboost/             # holdout, LOBO, importance, plots
│   ├── shap/                # SHAP importance, beeswarm, dependence, waterfall
│   └── evaluation/          # LOBO error analysis of the tuned Random Forest
├── scripts/                 # runnable pipeline steps (00 → 05)
├── src/                     # reusable modules imported by the scripts
├── run_all.py               # runs every step in order
├── requirements.txt
└── README.md
```

### `src/` modules

| Module | Responsibility |
|---|---|
| `config.py` | All paths, constants, model hyper-parameters |
| `data_processing.py` | Load `.mat` files, inspect cycles, build the cycle-level SoH dataset |
| `feature_engineering.py` | 24 candidate features, correlation analysis, automatic redundancy reduction |
| `battery_pipeline.py` | Orchestrates stages 1–4 and writes `data/processed/` |
| `model_utils.py` | Metrics, model factories, holdout/LOBO evaluation, shared plots |
| `evaluation.py` | LOBO error statistics, accuracy thresholds, error plots |

### Pipeline steps

| Step | Script | What it does |
|---|---|---|
| 00 | `scripts/00_build_dataset.py` | Raw `.mat` → cycle dataset → 24 features → correlation filter (\|r\| ≥ 0.90) |
| 01 | `scripts/01_EDA.py` | SoH distributions, degradation curves, correlations, feature plots |
| 02 | `scripts/02_RandomForest.py` | Depth sanity check, baseline vs tuned, holdout + LOBO, saves model |
| 03 | `scripts/03_XGBoost.py` | Baseline vs tuned, holdout + LOBO, saves model |
| 04 | `scripts/04_SHAP.py` | Global and local SHAP explanations for the final Random Forest |
| 05 | `scripts/05_model_evaluation.py` | LOBO error by battery / SoH range, accuracy thresholds, largest errors |

Each step reads the files written by the previous ones, so run them in order.