"""
INTERGEN - reviewer-revision experiment with fair architecture screening (v7)
==============================================

This single script replaces the separate collision.py and bike.py programs and
implements the reviewer-requested methodological checks that can be performed
with the two supplied datasets.

Key revisions implemented in code
---------------------------------
1. One script for both tasks via `collision` and `bike` switches.
2. Strict train / validation / test separation; preprocessing is fitted only on
   the training partition (no scaler leakage).
3. Task-appropriate metrics:
   - Classification: ROC-AUC, F1, precision, recall, balanced accuracy,
     accuracy, log loss and Brier score.
   - Regression: MSE, RMSE, MAE and R^2.
4. Reviewer-requested weight-space baselines:
   - naive equal-weight averaging,
   - aligned uniform Model Soup,
   - aligned greedy Model Soup,
   - one-step performance-weighted fusion,
   - Stochastic Weight Averaging (SWA),
   - Federated Averaging (FedAvg).
5. Permutation-symmetry handling through layer-wise neuron matching using the
   Hungarian algorithm, plus aligned-vs-unaligned ablation and a function-
   preservation diagnostic.
6. INTERGEN ablations:
   - recursive vs one-step fusion,
   - performance-based vs equal fusion coefficients,
   - best-worst / similar-performance / random pairing,
   - alternative score functions,
   - different numbers of base networks,
   - aligned vs unaligned weights.
7. More repeated runs (default = 20 seeds), 95% confidence intervals,
   paired Wilcoxon signed-rank tests, Holm correction and rank-biserial effect
   sizes.
8. Model-diversity, prediction-disagreement, parameter-distance, fusion-
   coefficient, inference-time, model-size and training-time outputs.
9. Reproducibility outputs: configuration, package versions, split summaries,
   preprocessing summaries and all raw result tables are written to disk.

Architecture-screening / fair-comparison note
---------------------------------------------
This version adds a `basit` switch. When `basit=True`, it runs a deliberately
small validation-only architecture screening designed to choose a sensible base
ANN architecture quickly without changing the INTERGEN algorithm. Neural
comparisons use the SAME architecture/training settings within a task. A custom
ANN-Bagging baseline uses bootstrap samples but the same Keras architecture,
optimizer, epoch budget and network count. The locked test set is not
evaluated at all in screening mode; architecture ranking/recommendation uses
validation performance only.

When `basit=False`, the full reviewer-facing experiment is run with the selected
architecture: 20 paired seeds, direct weight-fusion baselines, SWA, FedAvg,
ablations, Wilcoxon/Holm tests, effect sizes, diversity and compute outputs.

Important scope note
--------------------
The reviewers also requested experiments on larger external datasets such as
NYC Taxi or PeMS. Those datasets were not supplied with the current code, so
this script deliberately does not invent undocumented preprocessing for them.
The experiment framework is modular so an additional task loader can be added
later without changing the fusion/statistical-analysis machinery.

Dependencies
------------
pandas, numpy, scipy, scikit-learn, tensorflow, openpyxl, python-dotenv

The original task-specific hidden-layer activation sequences are retained to
keep the revision comparable to the submitted experiments. The classification
output layer is corrected to sigmoid, because binary_crossentropy requires a
probability output unless `from_logits=True` is explicitly used.
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

# Load repository-local .env without overriding variables supplied by the shell
# or by the isolated worker processes.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, t as student_t, wilcoxon

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

# GitHub-friendly runtime switches.
#
# Defaults preserve the submitted experiment: both tasks are enabled and the
# full reviewer-facing experiment is selected. Override without editing code:
#   INTERGEN_RUN_COLLISION=0/1
#   INTERGEN_RUN_BIKE=0/1
#   INTERGEN_BASIT=0/1
collision = os.environ.get("INTERGEN_RUN_COLLISION", "1") == "1"
bike = os.environ.get("INTERGEN_RUN_BIKE", "1") == "1"

# -------------------------------------------------------------------------
# FAST ARCHITECTURE SCREENING SWITCH
# -------------------------------------------------------------------------
# basit=True  -> very fast preliminary architecture comparison.
#                NOT intended for final manuscript tables/statistical claims.
# basit=False -> full reviewer-facing experiment with the selected architecture.
basit = os.environ.get("INTERGEN_BASIT", "0") == "1"
# Quick screening deliberately uses a tiny compute budget. Architecture
# selection is based ONLY on validation performance; the test set is never used
# to choose the architecture.
QUICK_REPEATS = 2
QUICK_NUM_NETWORKS = 4
QUICK_EPOCHS = 50
QUICK_SKLEARN_N_ESTIMATORS = 50
QUICK_RUN_ANN_BAGGING = True
QUICK_RUN_SWA = True
QUICK_RUN_CLASSICAL_REFERENCES = True

# Full reviewer experiment.
FINAL_REPEATS = 20
FINAL_NUM_NETWORKS = 16
FINAL_EPOCHS = 100
FINAL_SKLEARN_N_ESTIMATORS = 200

# Repository-relative defaults. Dataset files are intentionally not bundled.
# Paths can be overridden without editing the source:
#   INTERGEN_DATA_DIR
#   INTERGEN_COLLISION_FILE
#   INTERGEN_BIKE_FILE
#   INTERGEN_OUTPUT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(
    os.environ.get("INTERGEN_DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser()

COLLISION_FILE = Path(
    os.environ.get(
        "INTERGEN_COLLISION_FILE",
        str(DATA_DIR / "collision_2023_clevand.xlsx"),
    )
).expanduser()
BIKE_FILE = Path(
    os.environ.get(
        "INTERGEN_BIKE_FILE",
        str(DATA_DIR / "day_bike_share.xlsx"),
    )
).expanduser()

_DEFAULT_OUTPUT_NAME = (
    "intergen_architecture_screening_v7"
    if basit else
    "intergen_revision_final_fair_v7_results"
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "INTERGEN_OUTPUT_ROOT",
        str(PROJECT_ROOT / _DEFAULT_OUTPUT_NAME),
    )
).expanduser()

N_REPEATS = QUICK_REPEATS if basit else FINAL_REPEATS
_SINGLE_SEED_ENV = os.environ.get("INTERGEN_SINGLE_SEED")
SEEDS = (
    [int(_SINGLE_SEED_ENV)]
    if _SINGLE_SEED_ENV
    else [20260811 + i for i in range(N_REPEATS)]
)

# The fast screening is intentionally kept in one process because it is short
# and repeatedly changes the active architecture. The full experiment keeps the
# v4 per-seed process isolation/resume protection.
ISOLATE_REPEATS_IN_SUBPROCESS = False if basit else True
RESUME_COMPLETED_REPEATS = False if basit else True
KEEP_WORKER_OUTPUTS = True

NUM_NETWORKS = QUICK_NUM_NETWORKS if basit else FINAL_NUM_NETWORKS
NETWORK_COUNTS_ABLATION = tuple(n for n in (4, 8, 16) if n <= NUM_NETWORKS)
HIDDEN_UNITS = 16  # retained only for the submitted/original architecture
EPOCHS = QUICK_EPOCHS if basit else FINAL_EPOCHS
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Fixed-epoch training keeps the dedicated validation split out of base-model
# optimization. It is used only for fusion/model selection.
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.20

# The submitted code retrained each fused network for 100 extra epochs, whereas
# the manuscript algorithm describes weight fusion followed by re-evaluation.
# Default 0 isolates the effect of weight fusion and avoids unfair extra compute.
# Set >0 only for a clearly labelled sensitivity analysis.
POST_FUSION_FINETUNE_EPOCHS = 0

# Independent initializations preserve model diversity; neuron matching is then
# used before aligned fusion. Set True for a shared-initialization sensitivity
# analysis.
SHARED_INITIALIZATION = False

# Main INTERGEN settings.
DEFAULT_PAIRING = "best_worst"          # best_worst | similar | random
DEFAULT_ALIGNMENT = True
DEFAULT_SCORE_FUNCTION = {
    "collision": "auc",
    "bike": "r2_over_mse",            # preserves the submitted score idea
}

CLASSIFICATION_SCORE_FUNCTIONS = (
    "auc",
    "f1",
    "balanced_accuracy",
    "inverse_logloss",
)
REGRESSION_SCORE_FUNCTIONS = (
    "r2_over_mse",
    "inverse_mse",
    "inverse_mae",
    "r2_positive",
)
PAIRING_STRATEGIES = ("best_worst", "similar", "random")

# Expensive reviewer-requested components are OFF in the quick architecture
# screen and ON in the full final run. The quick screen has its own compact set
# of fair neural comparisons below.
RUN_WEIGHT_BASELINES = not basit
RUN_SKLEARN_BASELINES = not basit
RUN_ABLATIONS = not basit
RUN_STATISTICAL_TESTS = not basit
RUN_DIVERSITY_ANALYSIS = not basit
RUN_COMPUTE_BENCHMARKS = not basit

# Additional architecture-controlled output-level ensemble/bagging baselines.
# These are especially useful for a fair comparison with INTERGEN because they
# use the same Keras architecture and training budget.
RUN_DEEP_ENSEMBLE_ANN = True
RUN_ANN_BAGGING = True if not basit else QUICK_RUN_ANN_BAGGING

SAVE_FIRST_SEED_INTERGEN_MODEL = (
    False if basit else os.environ.get("INTERGEN_SAVE_MODEL", "1") == "1"
)

# SWA: average checkpoints from the last 40% of a single training trajectory.
SWA_START_FRACTION = 0.60

# FedAvg baseline. This is a standard synchronous data-weighted aggregation
# baseline, not INTERGEN. It intentionally uses a shared global initialization.
FEDAVG_CLIENTS = 8
FEDAVG_ROUNDS = 5
FEDAVG_LOCAL_EPOCHS = 5

SKLEARN_N_ESTIMATORS = QUICK_SKLEARN_N_ESTIMATORS if basit else FINAL_SKLEARN_N_ESTIMATORS
INFERENCE_REPEATS = 1 if basit else 3
PREDICT_BATCH_SIZE = 4096
# Keep this at 1. Some Keras/TensorFlow versions on Windows incorrectly exhaust
# finite NumPy-backed datasets when steps_per_execution > 1 and the number of
# batches is not an exact multiple of that value. That can silently truncate
# training. We therefore use the safe value 1 for the final experiments.
STEPS_PER_EXECUTION = 1

# Console-only live progress. These settings do NOT change training, splits,
# metrics, fusion, seeds, or any scientific result. They only make long runs
# visibly report what is happening.
LIVE_PROGRESS = True
PROGRESS_EVERY_EPOCHS = 20

# Classification decisions are NOT forced to 0.5 after weight fusion. A single
# threshold is selected on the held-out validation set for each fitted method
# and then frozen for train/test reporting. We still report 0.5-threshold
# diagnostics, so calibration shifts caused by weight fusion remain visible.
CLASSIFICATION_THRESHOLD_OBJECTIVE = "f1"  # f1 | balanced_accuracy
EPS = 1e-12

# The current collision file is already the Cleveland subset used in the
# submitted analysis. If a full STATS19 file is used later, add an explicit
# filtering rule in load_task_dataframe() and document it in the manuscript.
COLLISION_TARGET = "urban_or_rural_area"
BIKE_TARGET = "cnt"

# Daily bike-share data do NOT use an `hour` predictor. This resolves the
# inconsistency noted by Reviewer 2 between a daily dataset (731 rows) and the
# manuscript's mention of an hourly variable.
COLLISION_NUMERIC = [
    "number_of_vehicles",
    "number_of_casualties",
    "speed_limit",
]
COLLISION_CATEGORICAL = [
    "day_of_week",
    "first_road_class",
    "road_type",
    "junction_control",
    "pedestrian_crossing_human_control",
    "pedestrian_crossing_physical_facilities",
    "weather_conditions",
    "road_surface_conditions",
    "accident_severity",
]
BIKE_NUMERIC = ["temp", "atemp", "hum", "windspeed"]
BIKE_CATEGORICAL = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
]

# Original hidden-layer activation sequences retained from the submitted code.
COLLISION_ACTIVATIONS = [
    "sigmoid", "relu", "sigmoid", "relu", "relu", "relu", "sigmoid",
    "relu", "relu", "relu", "relu", "sigmoid", "relu", "sigmoid",
]
BIKE_ACTIVATIONS = [
    "relu", "relu", "relu", "relu", "relu", "sigmoid", "relu", "relu",
    "relu", "relu", "relu", "relu", "relu", "sigmoid", "relu", "relu",
    "relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu",
    "relu", "relu", "relu", "relu",
]


# -------------------------------------------------------------------------
# PRE-SPECIFIED ARCHITECTURE CANDIDATES
# -------------------------------------------------------------------------
# These are intentionally simple. The goal is NOT to invent a new INTERGEN;
# it is to test whether the weak bike regression result is caused by an
# unnecessarily deep base ANN. All neural methods within a task use the exact
# same selected architecture, so differences reflect ensemble/fusion strategy
# rather than hidden-layer capacity.
#
# Each layer is (units, activation).
ARCHITECTURE_LIBRARY: Dict[str, Dict[str, List[Tuple[int, str]]]] = {
    "collision": {
        "shallow_2x32_relu": [(32, "relu"), (32, "relu")],
        "medium_3x32_relu": [(32, "relu"), (32, "relu"), (32, "relu")],
        "medium_4x32_relu": [(32, "relu"), (32, "relu"), (32, "relu"), (32, "relu")],
        "submitted_original": [(HIDDEN_UNITS, a) for a in COLLISION_ACTIVATIONS],
    },
    "bike": {
        "shallow_2x32_relu": [(32, "relu"), (32, "relu")],
        "medium_3x32_relu": [(32, "relu"), (32, "relu"), (32, "relu")],
        "medium_4x32_relu": [(32, "relu"), (32, "relu"), (32, "relu"), (32, "relu")],
        "medium_5x32_relu": [
            (32, "relu"),
            (32, "relu"),
            (32, "relu"),
            (32, "relu"),
            (32, "relu"),
        ],
        "submitted_original": [(HIDDEN_UNITS, a) for a in BIKE_ACTIVATIONS],
    },
}

SCREEN_ARCHITECTURES: Dict[str, Tuple[str, ...]] = {
    "collision": (
        "shallow_2x32_relu",
        "medium_3x32_relu",
        "medium_4x32_relu",
        "submitted_original",
    ),
    "bike": (
        "shallow_2x32_relu",
        "medium_3x32_relu",
        "medium_4x32_relu",
        "medium_5x32_relu",
        "submitted_original",
    ),
}

# After the basit=True screen, set these two names to the validation-selected
# architectures before running with basit=False. They start at the submitted
# architectures so the script never silently changes the manuscript design.
# FINAL ARCHITECTURES FROZEN AFTER VALIDATION-ONLY SCREENING:
# - Collision: submitted/original architecture.
# - Bike: 5 hidden layers x 32 ReLU units.
# The test set was not used to choose these architectures.
FINAL_ARCHITECTURE: Dict[str, str] = {
    "collision": os.environ.get(
        "INTERGEN_COLLISION_ARCH", "submitted_original"
    ),
    "bike": os.environ.get(
        "INTERGEN_BIKE_ARCH", "medium_5x32_relu"
    ),
}

# Mutable only inside the quick screen; full reviewer runs use FINAL_ARCHITECTURE.
ACTIVE_ARCHITECTURE: Dict[str, str] = dict(FINAL_ARCHITECTURE)


def architecture_layers(task: str, architecture_name: Optional[str] = None) -> List[Tuple[int, str]]:
    name = architecture_name or ACTIVE_ARCHITECTURE[task]
    if task not in ARCHITECTURE_LIBRARY or name not in ARCHITECTURE_LIBRARY[task]:
        raise ValueError(f"Unknown architecture for {task}: {name}")
    return list(ARCHITECTURE_LIBRARY[task][name])


def architecture_description(task: str, architecture_name: Optional[str] = None) -> str:
    name = architecture_name or ACTIVE_ARCHITECTURE[task]
    spec = architecture_layers(task, name)
    body = " -> ".join(f"{units}:{activation}" for units, activation in spec)
    return f"{name} [{body}]"


# =============================================================================
# 1. REPRODUCIBILITY / UTILITIES
# =============================================================================


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and TensorFlow RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def enable_determinism() -> None:
    """Request deterministic TensorFlow operations when supported."""
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_output_dirs(task: str) -> Dict[str, Path]:
    base = OUTPUT_ROOT / task
    paths = {
        "base": base,
        "raw": base / "raw",
        "summary": base / "summary",
        "models": base / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibility across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_dense_float32(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def safe_json_value(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=safe_json_value)


def environment_info() -> Dict[str, Any]:
    import sklearn
    import scipy

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "tensorflow": tf.__version__,
    }


# =============================================================================
# 2. DATA LOADING / PREPROCESSING
# =============================================================================


@dataclass
class TaskData:
    task: str
    df: pd.DataFrame
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    target_column: str
    target_note: str


def _encode_collision_target(series: pd.Series) -> Tuple[pd.Series, str]:
    """
    Convert collision target to 0/1 without silently guessing when ambiguous.

    Preferred convention for STATS19-coded data:
      1 = Urban  -> positive class 1
      2 = Rural  -> negative class 0
      other/unallocated codes are dropped.
    String Urban/Rural labels are handled as well.
    """
    s = series.copy()

    # Try numeric STATS19 coding first.
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_nonmissing = numeric.dropna()
    if len(numeric_nonmissing) > 0:
        unique_num = set(np.unique(numeric_nonmissing).tolist())
        if 1 in unique_num and 2 in unique_num and unique_num.issubset({1, 2, 3, 9}):
            mapped = numeric.map({1: 1.0, 2: 0.0})
            note = (
                "Collision target mapping: STATS19 code 1 (Urban) -> 1; "
                "code 2 (Rural) -> 0; other/unallocated target codes dropped."
            )
            return mapped, note

    # String labels.
    lower = s.astype(str).str.strip().str.lower()
    if lower.str.contains("urban", na=False).any() and lower.str.contains("rural", na=False).any():
        mapped = pd.Series(np.nan, index=s.index, dtype=float)
        mapped[lower.str.contains("urban", na=False)] = 1.0
        mapped[lower.str.contains("rural", na=False)] = 0.0
        note = "Collision target mapping: Urban -> 1; Rural -> 0; other labels dropped."
        return mapped, note

    # Generic binary fallback only if exactly two observed labels exist.
    vals = [v for v in pd.unique(s.dropna())]
    if len(vals) == 2:
        ordered = sorted(vals, key=lambda z: str(z))
        mapping = {ordered[0]: 0.0, ordered[1]: 1.0}
        return s.map(mapping), f"Generic binary target mapping used: {mapping}"

    raise ValueError(
        "urban_or_rural_area could not be mapped safely to a binary target. "
        "Please inspect its unique values and define an explicit mapping."
    )


def load_task_dataframe(task: str) -> TaskData:
    if task == "collision":
        path = COLLISION_FILE
        if not path.exists():
            raise FileNotFoundError(f"Collision file not found: {path}")
        df = pd.read_excel(path)
        features = COLLISION_NUMERIC + COLLISION_CATEGORICAL
        required = features + [COLLISION_TARGET]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Collision data are missing columns: {missing_cols}")

        y_binary, target_note = _encode_collision_target(df[COLLISION_TARGET])
        df = df.copy()
        df[COLLISION_TARGET] = y_binary
        before = len(df)
        df = df[df[COLLISION_TARGET].notna()].copy()
        df[COLLISION_TARGET] = df[COLLISION_TARGET].astype(int)
        dropped = before - len(df)
        target_note += f" Rows dropped for invalid/missing target: {dropped}."

        return TaskData(
            task=task,
            df=df,
            feature_columns=features,
            numeric_columns=COLLISION_NUMERIC,
            categorical_columns=COLLISION_CATEGORICAL,
            target_column=COLLISION_TARGET,
            target_note=target_note,
        )

    if task == "bike":
        path = BIKE_FILE
        if not path.exists():
            raise FileNotFoundError(f"Bike-share file not found: {path}")
        df = pd.read_excel(path)
        features = BIKE_NUMERIC + BIKE_CATEGORICAL
        required = features + [BIKE_TARGET]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Bike-share data are missing columns: {missing_cols}")

        df = df.copy()
        df[BIKE_TARGET] = pd.to_numeric(df[BIKE_TARGET], errors="coerce")
        before = len(df)
        df = df[df[BIKE_TARGET].notna()].copy()
        dropped = before - len(df)
        target_note = (
            f"Regression target cnt retained on original scale. Rows dropped for "
            f"invalid/missing target: {dropped}. The daily dataset intentionally excludes hour."
        )
        return TaskData(
            task=task,
            df=df,
            feature_columns=features,
            numeric_columns=BIKE_NUMERIC,
            categorical_columns=BIKE_CATEGORICAL,
            target_column=BIKE_TARGET,
            target_note=target_note,
        )

    raise ValueError(f"Unknown task: {task}")


def split_dataframe(task_data: TaskData, seed: int) -> Dict[str, Any]:
    X = task_data.df[task_data.feature_columns].copy()
    y = task_data.df[task_data.target_column].to_numpy()

    stratify = y if task_data.task == "collision" else None
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=seed,
        stratify=stratify,
    )

    # VALIDATION_FRACTION is defined relative to the full dataset.
    val_relative_to_dev = VALIDATION_FRACTION / (1.0 - TEST_FRACTION)
    stratify_dev = y_dev if task_data.task == "collision" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=val_relative_to_dev,
        random_state=seed + 17,
        stratify=stratify_dev,
    )

    return {
        "X_train_df": X_train,
        "X_val_df": X_val,
        "X_test_df": X_test,
        "y_train": np.asarray(y_train),
        "y_val": np.asarray(y_val),
        "y_test": np.asarray(y_test),
    }


def build_preprocessor(task_data: TaskData) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, task_data.numeric_columns),
            ("categorical", categorical_pipe, task_data.categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def preprocess_split(task_data: TaskData, split: Dict[str, Any]) -> Dict[str, Any]:
    preprocessor = build_preprocessor(task_data)
    X_train = to_dense_float32(preprocessor.fit_transform(split["X_train_df"]))
    X_val = to_dense_float32(preprocessor.transform(split["X_val_df"]))
    X_test = to_dense_float32(preprocessor.transform(split["X_test_df"]))

    try:
        output_features = preprocessor.get_feature_names_out().tolist()
    except Exception:
        output_features = [f"feature_{i}" for i in range(X_train.shape[1])]

    return {
        **split,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "output_features": output_features,
    }


def data_summary(task_data: TaskData, processed: Dict[str, Any], seed: int) -> Dict[str, Any]:
    df = task_data.df
    missing_by_column = df[task_data.feature_columns + [task_data.target_column]].isna().sum().to_dict()
    summary: Dict[str, Any] = {
        "task": task_data.task,
        "seed": seed,
        "rows_after_target_cleaning": len(df),
        "raw_feature_count": len(task_data.feature_columns),
        "transformed_feature_count": processed["X_train"].shape[1],
        "train_n": len(processed["y_train"]),
        "validation_n": len(processed["y_val"]),
        "test_n": len(processed["y_test"]),
        "target_note": task_data.target_note,
        "missing_values_before_imputation": missing_by_column,
        "transformed_features": processed["output_features"],
    }
    if task_data.task == "collision":
        for split_name in ("train", "val", "test"):
            y = processed[f"y_{split_name}"]
            values, counts = np.unique(y, return_counts=True)
            summary[f"{split_name}_class_distribution"] = {
                str(int(v)): int(c) for v, c in zip(values, counts)
            }
    else:
        for split_name in ("train", "val", "test"):
            y = processed[f"y_{split_name}"]
            summary[f"{split_name}_target_mean"] = float(np.mean(y))
            summary[f"{split_name}_target_std"] = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    return summary


# =============================================================================
# 3. MODEL ARCHITECTURE / TRAINING
# =============================================================================


def create_model(
    task: str,
    input_dim: int,
    architecture_name: Optional[str] = None,
) -> tf.keras.Model:
    """Create the active architecture for this task.

    Architecture is the ONLY element changed by the quick screen. The INTERGEN
    fusion algorithm, optimizer, loss, preprocessing and split protocol remain
    unchanged.
    """
    spec = architecture_layers(task, architecture_name)
    name = architecture_name or ACTIVE_ARCHITECTURE[task]

    layers: List[tf.keras.layers.Layer] = [Input(shape=(input_dim,), name="features")]
    for i, (units, activation) in enumerate(spec, start=1):
        layers.append(
            Dense(int(units), activation=activation, name=f"hidden_{i:02d}")
        )

    if task == "collision":
        layers.append(Dense(1, activation="sigmoid", name="output"))
    else:
        layers.append(Dense(1, activation="linear", name="output"))

    model = Sequential(layers, name=f"INTERGEN_{task}_{name}")
    compile_model(model, task)
    return model


def compile_model(model: tf.keras.Model, task: str) -> None:
    """Compile conservatively for cross-version Keras/TensorFlow reliability."""
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy" if task == "collision" else "mean_squared_error",
        steps_per_execution=STEPS_PER_EXECUTION,
    )


def progress(message: str) -> None:
    """Print an immediately flushed timestamped progress message."""
    if LIVE_PROGRESS:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class EpochProgressCallback(tf.keras.callbacks.Callback):
    """Lightweight console heartbeat; it does not modify model state."""

    def __init__(self, label: str, total_epochs: int, every: int = PROGRESS_EVERY_EPOCHS):
        super().__init__()
        self.label = label
        self.total_epochs = int(total_epochs)
        self.every = max(1, int(every))
        self.started = 0.0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        done = epoch + 1
        if done == 1 or done % self.every == 0 or done == self.total_epochs:
            elapsed = time.perf_counter() - self.started
            loss = None if logs is None else logs.get("loss")
            loss_text = "" if loss is None else f" | loss={float(loss):.6g}"
            progress(
                f"{self.label}: epoch {done}/{self.total_epochs}"
                f" | elapsed={elapsed:.1f}s{loss_text}"
            )


def train_model(
    model: tf.keras.Model,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    seed: int,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    progress_label: Optional[str] = None,
) -> Tuple[tf.keras.Model, float]:
    set_global_seed(seed)
    t0 = time.perf_counter()
    active_callbacks = list(callbacks or [])
    if LIVE_PROGRESS and progress_label:
        active_callbacks.append(EpochProgressCallback(progress_label, epochs))
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=True,
        callbacks=active_callbacks,
    )
    completed_epochs = len(history.history.get("loss", []))
    if completed_epochs != epochs:
        raise RuntimeError(
            f"Training ended early: requested {epochs} epochs but completed "
            f"{completed_epochs}. Results from this seed are not valid."
        )
    elapsed = time.perf_counter() - t0
    return model, elapsed


def clone_model_with_weights(task: str, input_dim: int, weights: Sequence[np.ndarray]) -> tf.keras.Model:
    model = create_model(task, input_dim)
    model.set_weights([np.array(w, copy=True) for w in weights])
    return model


def train_base_models(
    task: str,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    experiment_seed: int,
) -> Tuple[List[tf.keras.Model], List[float], List[int]]:
    models: List[tf.keras.Model] = []
    train_times: List[float] = []
    model_seeds: List[int] = []

    shared_weights: Optional[List[np.ndarray]] = None
    if SHARED_INITIALIZATION:
        set_global_seed(experiment_seed + 999)
        template = create_model(task, input_dim)
        shared_weights = [np.array(w, copy=True) for w in template.get_weights()]
        del template

    progress(f"Base ANN training starts: {NUM_NETWORKS} models x {EPOCHS} epochs")
    base_started = time.perf_counter()
    for i in range(NUM_NETWORKS):
        model_seed = experiment_seed * 100 + i + 1
        progress(f"Base ANN {i + 1}/{NUM_NETWORKS} starting | model_seed={model_seed}")
        model_seeds.append(model_seed)
        set_global_seed(model_seed)
        model = create_model(task, input_dim)
        if shared_weights is not None:
            model.set_weights([np.array(w, copy=True) for w in shared_weights])
        model, elapsed = train_model(
            model,
            task,
            X_train,
            y_train,
            epochs=EPOCHS,
            seed=model_seed,
            progress_label=f"Base ANN {i + 1}/{NUM_NETWORKS}",
        )
        models.append(model)
        train_times.append(elapsed)
        progress(
            f"Base ANN {i + 1}/{NUM_NETWORKS} completed in {elapsed:.1f}s "
            f"| base total={sum(train_times) / 60.0:.1f} min"
        )

    progress(f"All {NUM_NETWORKS} base ANNs completed in {(time.perf_counter() - base_started) / 60.0:.1f} min")
    return models, train_times, model_seeds



# =============================================================================
# 3B. FAIR ANN ENSEMBLE BASELINES
# =============================================================================


def _ensemble_prediction(models: Sequence[tf.keras.Model], X: np.ndarray) -> np.ndarray:
    """Mean prediction/probability across same-architecture Keras models."""
    if not models:
        raise ValueError("Ensemble must contain at least one model.")
    preds = np.vstack([predict_keras(m, X) for m in models])
    return np.mean(preds, axis=0)


def evaluate_keras_ensemble_splits(
    models: Sequence[tf.keras.Model],
    task: str,
    processed: Dict[str, Any],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    predictions = {
        "train": _ensemble_prediction(models, processed["X_train"]),
        "val": _ensemble_prediction(models, processed["X_val"]),
        "test": _ensemble_prediction(models, processed["X_test"]),
    }
    y_map = {
        "train": processed["y_train"],
        "val": processed["y_val"],
        "test": processed["y_test"],
    }

    if task == "collision":
        threshold = select_classification_threshold(
            y_map["val"], predictions["val"], CLASSIFICATION_THRESHOLD_OBJECTIVE
        )
        result["decision_threshold"] = float(threshold)
        result["threshold_objective"] = CLASSIFICATION_THRESHOLD_OBJECTIVE
        for split_name in ("train", "val", "test"):
            metrics = classification_metrics(
                y_map[split_name], predictions[split_name], threshold=threshold
            )
            for k, v in metrics.items():
                result[f"{split_name}_{k}"] = float(v)
            fixed = classification_metrics(
                y_map[split_name], predictions[split_name], threshold=0.5
            )
            for k in (
                "accuracy", "balanced_accuracy", "precision", "recall",
                "f1", "predicted_positive_rate",
            ):
                result[f"{split_name}_{k}_at_0_5"] = float(fixed[k])
        result["generalization_gap_auc"] = (
            result["train_roc_auc"] - result["test_roc_auc"]
        )
        result["generalization_gap_logloss"] = (
            result["test_log_loss"] - result["train_log_loss"]
        )
        return result

    for split_name in ("train", "val", "test"):
        metrics = regression_metrics(y_map[split_name], predictions[split_name])
        for k, v in metrics.items():
            result[f"{split_name}_{k}"] = float(v)
    result["generalization_gap_r2"] = result["train_r2"] - result["test_r2"]
    result["generalization_gap_rmse"] = result["test_rmse"] - result["train_rmse"]
    return result


def keras_ensemble_inference_ms_per_sample(
    models: Sequence[tf.keras.Model],
    X: np.ndarray,
) -> float:
    if len(X) == 0:
        return np.nan
    _ = _ensemble_prediction(models, X[: min(16, len(X))])
    t0 = time.perf_counter()
    for _ in range(INFERENCE_REPEATS):
        _ = _ensemble_prediction(models, X)
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / (INFERENCE_REPEATS * len(X))


def method_record_keras_ensemble(
    task: str,
    seed: int,
    method: str,
    models: Sequence[tf.keras.Model],
    processed: Dict[str, Any],
    fit_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = evaluate_keras_ensemble_splits(models, task, processed)
    rec: Dict[str, Any] = {
        "task": task,
        "seed": seed,
        "method": method,
        "fit_seconds": float(fit_seconds),
        "model_params": int(sum(m.count_params() for m in models)),
        "model_size_bytes": int(sum(keras_model_size_bytes(m) for m in models)),
        "architecture": ACTIVE_ARCHITECTURE[task],
        "architecture_description": architecture_description(task),
        "ensemble_members": len(models),
        "deployable_models": len(models),
    }
    if RUN_COMPUTE_BENCHMARKS:
        rec["inference_ms_per_sample"] = keras_ensemble_inference_ms_per_sample(
            models, processed["X_test"]
        )
    rec.update(metrics)
    if extra:
        rec.update(extra)
    return rec


def train_ann_bagging(
    task: str,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_models: Optional[int] = None,
) -> Tuple[List[tf.keras.Model], List[float]]:
    """
    Architecture-controlled ANN bagging baseline.

    Each member uses the SAME Keras architecture, optimizer, learning rate,
    epoch budget and batch size as INTERGEN. The only bagging-specific change is
    bootstrap resampling of the training rows.
    """
    n_models = int(NUM_NETWORKS if n_models is None else n_models)
    rng = np.random.default_rng(seed)
    models: List[tf.keras.Model] = []
    times: List[float] = []

    progress(
        f"ANN_Bagging training starts: {n_models} bootstrap models x {EPOCHS} epochs "
        f"| architecture={ACTIVE_ARCHITECTURE[task]}"
    )
    n = len(y_train)
    for i in range(n_models):
        # Standard bootstrap sample, same size as the original training set.
        idx = rng.integers(0, n, size=n)
        if task == "collision":
            # Extremely unlikely on this dataset, but ensure both classes exist.
            attempts = 0
            while len(np.unique(y_train[idx])) < 2 and attempts < 20:
                idx = rng.integers(0, n, size=n)
                attempts += 1

        model_seed = seed * 100 + i + 1
        set_global_seed(model_seed)
        model = create_model(task, input_dim)
        model, elapsed = train_model(
            model,
            task,
            X_train[idx],
            y_train[idx],
            epochs=EPOCHS,
            seed=model_seed,
            progress_label=(
                f"ANN Bagging {i + 1}/{n_models}" if LIVE_PROGRESS and not basit else None
            ),
        )
        models.append(model)
        times.append(elapsed)
        progress(
            f"ANN_Bagging member {i + 1}/{n_models} completed in {elapsed:.1f}s"
        )

    return models, times


# =============================================================================
# 4. METRICS / FUSION SCORE
# =============================================================================


def predict_keras(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Fast direct Keras inference.

    Calling model.predict() repeatedly creates tf.data/MapDataset machinery and
    was a major source of warning spam and runtime growth in the original
    revision script. Direct eager-to-graph model calls avoid that overhead.
    """
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.asarray([], dtype=float)

    out: List[np.ndarray] = []
    for start in range(0, len(X), PREDICT_BATCH_SIZE):
        xb = tf.convert_to_tensor(X[start:start + PREDICT_BATCH_SIZE], dtype=tf.float32)
        yb = model(xb, training=False)
        out.append(np.asarray(yb).reshape(-1))
    return np.concatenate(out).astype(float, copy=False)


def select_classification_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    objective: str = CLASSIFICATION_THRESHOLD_OBJECTIVE,
) -> float:
    """
    Select a decision threshold using ONLY the validation set.

    This addresses the observed high-AUC/F1=0 pattern after weight fusion:
    ranking may remain strong while sigmoid probabilities shift below 0.5.
    The selected validation threshold is frozen before test evaluation.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)

    if len(np.unique(y_true)) < 2 or len(prob) == 0:
        return 0.5

    if objective == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, prob)
        if len(thresholds) == 0:
            return 0.5
        f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], EPS
        )
        best = np.flatnonzero(np.isclose(f1_values, np.nanmax(f1_values), rtol=1e-10, atol=1e-12))
        if len(best) == 0:
            return 0.5
        # Stable tie-break: prefer the equally good threshold closest to 0.5.
        idx = int(best[np.argmin(np.abs(thresholds[best] - 0.5))])
        return float(np.clip(thresholds[idx], EPS, 1.0 - EPS))

    if objective == "balanced_accuracy":
        unique = np.unique(prob)
        if len(unique) == 1:
            return float(unique[0])
        candidates = np.concatenate(
            ([EPS], (unique[:-1] + unique[1:]) / 2.0, [1.0 - EPS])
        )
        scores = np.asarray([
            balanced_accuracy_score(y_true, (prob >= t).astype(int))
            for t in candidates
        ])
        best = np.flatnonzero(np.isclose(scores, np.nanmax(scores), rtol=1e-10, atol=1e-12))
        idx = int(best[np.argmin(np.abs(candidates[best] - 0.5))])
        return float(candidates[idx])

    raise ValueError(f"Unknown threshold objective: {objective}")


def classification_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    prob = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
    y_true = np.asarray(y_true, dtype=int)
    pred = (prob >= float(threshold)).astype(int)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": auc,
        "log_loss": log_loss(y_true, prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, prob),
        "predicted_positive_rate": float(np.mean(pred)),
        "probability_mean": float(np.mean(prob)),
        "probability_min": float(np.min(prob)),
        "probability_max": float(np.max(prob)),
        "mean_probability_true_positive": float(np.mean(prob[pos_mask])) if np.any(pos_mask) else np.nan,
        "mean_probability_true_negative": float(np.mean(prob[neg_mask])) if np.any(neg_mask) else np.nan,
    }


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mse = mean_squared_error(y_true, pred)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def metrics_from_predictions(task: str, y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Metrics for validation-driven model/fusion scoring.

    For classification, threshold-dependent validation scores (F1/balanced
    accuracy) use a threshold selected on that same validation set. This is
    appropriate for model selection; final test evaluation uses a separately
    frozen validation threshold in evaluate_*_splits().
    """
    if task == "collision":
        threshold = select_classification_threshold(y_true, pred)
        m = classification_metrics(y_true, pred, threshold=threshold)
        m["decision_threshold"] = threshold
        return m
    return regression_metrics(y_true, pred)


def fusion_score(task: str, metrics: Dict[str, float], score_name: str) -> float:
    """Return a non-negative score used only for fusion/model ordering."""
    if task == "collision":
        if score_name == "auc":
            value = metrics["roc_auc"]
        elif score_name == "f1":
            value = metrics["f1"]
        elif score_name == "balanced_accuracy":
            value = metrics["balanced_accuracy"]
        elif score_name == "inverse_logloss":
            value = 1.0 / max(metrics["log_loss"], EPS)
        else:
            raise ValueError(f"Unknown classification score function: {score_name}")
    else:
        mse = max(metrics["mse"], EPS)
        mae = max(metrics["mae"], EPS)
        r2 = metrics["r2"]
        if score_name == "r2_over_mse":
            value = max(r2, 0.0) / mse
        elif score_name == "inverse_mse":
            value = 1.0 / mse
        elif score_name == "inverse_mae":
            value = 1.0 / mae
        elif score_name == "r2_positive":
            value = max(r2, 0.0)
        else:
            raise ValueError(f"Unknown regression score function: {score_name}")

    if not np.isfinite(value) or value < 0:
        return EPS
    return float(value + EPS)


def evaluate_keras_splits(
    model: tf.keras.Model,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    predictions = {
        "train": predict_keras(model, X_train),
        "val": predict_keras(model, X_val),
        "test": predict_keras(model, X_test),
    }
    y_map = {"train": y_train, "val": y_val, "test": y_test}

    if task == "collision":
        # Choose once on validation, then freeze. No test-set threshold tuning.
        threshold = select_classification_threshold(
            y_val, predictions["val"], CLASSIFICATION_THRESHOLD_OBJECTIVE
        )
        result["decision_threshold"] = float(threshold)
        result["threshold_objective"] = CLASSIFICATION_THRESHOLD_OBJECTIVE

        for split_name in ("train", "val", "test"):
            y = y_map[split_name]
            prob = predictions[split_name]
            metrics = classification_metrics(y, prob, threshold=threshold)
            for k, v in metrics.items():
                result[f"{split_name}_{k}"] = float(v)

            # Transparent fixed-0.5 diagnostics: these show whether fusion caused
            # a probability-scale/calibration shift even when AUC remains high.
            fixed = classification_metrics(y, prob, threshold=0.5)
            for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "predicted_positive_rate"):
                result[f"{split_name}_{k}_at_0_5"] = float(fixed[k])

        result["generalization_gap_auc"] = result["train_roc_auc"] - result["test_roc_auc"]
        result["generalization_gap_logloss"] = result["test_log_loss"] - result["train_log_loss"]
        return result

    for split_name in ("train", "val", "test"):
        metrics = regression_metrics(y_map[split_name], predictions[split_name])
        for k, v in metrics.items():
            result[f"{split_name}_{k}"] = float(v)

    result["generalization_gap_r2"] = result["train_r2"] - result["test_r2"]
    result["generalization_gap_rmse"] = result["test_rmse"] - result["train_rmse"]
    return result


def validation_metrics_for_models(
    models: Sequence[tf.keras.Model],
    task: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> List[Dict[str, float]]:
    return [metrics_from_predictions(task, y_val, predict_keras(m, X_val)) for m in models]


# =============================================================================
# 5. PERMUTATION ALIGNMENT
# =============================================================================


def _neuron_feature_matrix(kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # Each row describes one neuron by incoming weights + bias.
    features = np.concatenate([kernel.T, bias.reshape(-1, 1)], axis=1).astype(float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, EPS)


def align_weight_list_to_reference(
    reference_weights: Sequence[np.ndarray],
    candidate_weights: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """
    Layer-wise neuron matching for same-architecture dense networks.

    The candidate network is permuted, not functionally changed. For each hidden
    layer, the Hungarian algorithm matches normalized incoming-weight+bias
    vectors to the reference. The same permutation is applied to the rows of the
    next layer's kernel, preserving the candidate network's input-output map.
    """
    ref = [np.array(w, copy=False) for w in reference_weights]
    cand = [np.array(w, copy=True) for w in candidate_weights]

    if len(ref) != len(cand) or len(ref) % 2 != 0:
        raise ValueError("Weight lists are incompatible for dense-layer alignment.")
    for a, b in zip(ref, cand):
        if a.shape != b.shape:
            raise ValueError(
                "Permutation alignment requires architecture-compatible models "
                f"with identical weight shapes, got {a.shape} vs {b.shape}."
            )

    n_dense_layers = len(ref) // 2
    # Do not permute the scalar output layer; align hidden layers only.
    for layer_idx in range(n_dense_layers - 1):
        k = 2 * layer_idx
        next_k = 2 * (layer_idx + 1)

        W_ref, b_ref = ref[k], ref[k + 1]
        W_cand, b_cand = cand[k], cand[k + 1]

        f_ref = _neuron_feature_matrix(W_ref, b_ref)
        f_cand = _neuron_feature_matrix(W_cand, b_cand)
        # Cosine-distance assignment.
        cost = 1.0 - (f_ref @ f_cand.T)
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = col_ind[np.argsort(row_ind)]

        cand[k] = W_cand[:, perm]
        cand[k + 1] = b_cand[perm]
        # Propagate the same neuron permutation into outgoing connections.
        cand[next_k] = cand[next_k][perm, :]

    return cand


def alignment_function_preservation_error(
    task: str,
    input_dim: int,
    reference_model: tf.keras.Model,
    candidate_model: tf.keras.Model,
    X: np.ndarray,
) -> float:
    original_pred = predict_keras(candidate_model, X)
    aligned_weights = align_weight_list_to_reference(
        reference_model.get_weights(), candidate_model.get_weights()
    )
    aligned_model = clone_model_with_weights(task, input_dim, aligned_weights)
    aligned_pred = predict_keras(aligned_model, X)
    error = float(np.max(np.abs(original_pred - aligned_pred)))
    del aligned_model
    return error


# =============================================================================
# 6. WEIGHT FUSION OPERATORS
# =============================================================================


def normalize_coefficients(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= EPS:
        return np.full(len(arr), 1.0 / len(arr), dtype=float)
    return arr / total


def average_weight_lists(weight_lists: Sequence[Sequence[np.ndarray]], coefficients: Sequence[float]) -> List[np.ndarray]:
    coeff = normalize_coefficients(coefficients)
    if len(weight_lists) != len(coeff):
        raise ValueError("Number of models and coefficients must match.")
    result: List[np.ndarray] = []
    for layer_weights in zip(*weight_lists):
        out = np.zeros_like(layer_weights[0], dtype=np.float64)
        for c, w in zip(coeff, layer_weights):
            out += c * np.asarray(w, dtype=np.float64)
        result.append(out.astype(layer_weights[0].dtype, copy=False))
    return result


def one_step_fusion(
    models: Sequence[tf.keras.Model],
    val_metrics: Sequence[Dict[str, float]],
    task: str,
    input_dim: int,
    score_name: str,
    alignment: bool,
    weighting: str,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    scores = [fusion_score(task, m, score_name) for m in val_metrics]
    ref_idx = int(np.argmax(scores))
    ref_weights = models[ref_idx].get_weights()

    weight_lists: List[List[np.ndarray]] = []
    for i, model in enumerate(models):
        w = model.get_weights()
        if alignment and i != ref_idx:
            w = align_weight_list_to_reference(ref_weights, w)
        else:
            w = [np.array(x, copy=True) for x in w]
        weight_lists.append(w)

    if weighting == "performance":
        coeff = normalize_coefficients(scores)
    elif weighting == "equal":
        coeff = np.full(len(models), 1.0 / len(models))
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    fused_weights = average_weight_lists(weight_lists, coeff)
    model = clone_model_with_weights(task, input_dim, fused_weights)
    meta = {
        "reference_index": ref_idx,
        "alignment": alignment,
        "weighting": weighting,
        "score_function": score_name,
        "coefficients": coeff.tolist(),
    }
    return model, meta


def pairing_indices(scores: Sequence[float], strategy: str, rng: np.random.Generator) -> Tuple[List[Tuple[int, int]], List[int]]:
    n = len(scores)
    if n < 2:
        return [], list(range(n))

    if strategy == "best_worst":
        order = list(np.argsort(-np.asarray(scores)))
        pairs: List[Tuple[int, int]] = []
        carries: List[int] = []
        left, right = 0, n - 1
        while left < right:
            pairs.append((order[left], order[right]))
            left += 1
            right -= 1
        if left == right:
            carries.append(order[left])
        return pairs, carries

    if strategy == "similar":
        order = list(np.argsort(-np.asarray(scores)))
    elif strategy == "random":
        order = list(rng.permutation(n))
    else:
        raise ValueError(f"Unknown pairing strategy: {strategy}")

    pairs = [(order[i], order[i + 1]) for i in range(0, n - 1, 2)]
    carries = [order[-1]] if n % 2 else []
    return pairs, carries


def fuse_two_models(
    model_a: tf.keras.Model,
    model_b: tf.keras.Model,
    score_a: float,
    score_b: float,
    task: str,
    input_dim: int,
    alignment: bool,
    weighting: str,
) -> Tuple[tf.keras.Model, float, float, str]:
    # Use the higher-scoring model as the alignment reference.
    if score_b > score_a:
        model_a, model_b = model_b, model_a
        score_a, score_b = score_b, score_a
        swapped = "yes"
    else:
        swapped = "no"

    wa = [np.array(w, copy=True) for w in model_a.get_weights()]
    wb = [np.array(w, copy=True) for w in model_b.get_weights()]
    if alignment:
        wb = align_weight_list_to_reference(wa, wb)

    if weighting == "performance":
        alpha_a, alpha_b = normalize_coefficients([score_a, score_b])
    elif weighting == "equal":
        alpha_a, alpha_b = 0.5, 0.5
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    fused_weights = average_weight_lists([wa, wb], [alpha_a, alpha_b])
    fused = clone_model_with_weights(task, input_dim, fused_weights)
    return fused, float(alpha_a), float(alpha_b), swapped


def recursive_intergen(
    models: Sequence[tf.keras.Model],
    task: str,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    score_name: str,
    pairing: str,
    alignment: bool,
    weighting: str,
    seed: int,
    names: Optional[Sequence[str]] = None,
) -> Tuple[tf.keras.Model, List[Dict[str, Any]]]:
    if names is None:
        names = [f"ANN{i+1:02d}" for i in range(len(models))]

    pool: List[Dict[str, Any]] = []
    for name, model in zip(names, models):
        vm = metrics_from_predictions(task, y_val, predict_keras(model, X_val))
        pool.append(
            {
                "name": name,
                "model": model,
                "val_metrics": vm,
                "score": fusion_score(task, vm, score_name),
            }
        )

    rng = np.random.default_rng(seed)
    history: List[Dict[str, Any]] = []
    stage = 0

    while len(pool) > 1:
        stage += 1
        scores = [p["score"] for p in pool]
        pairs, carries = pairing_indices(scores, pairing, rng)
        new_pool: List[Dict[str, Any]] = []

        for pair_number, (i, j) in enumerate(pairs, start=1):
            a, b = pool[i], pool[j]
            fused, alpha_ref, alpha_other, swapped = fuse_two_models(
                a["model"],
                b["model"],
                a["score"],
                b["score"],
                task,
                input_dim,
                alignment,
                weighting,
            )

            finetune_seconds = 0.0
            if POST_FUSION_FINETUNE_EPOCHS > 0:
                fused, finetune_seconds = train_model(
                    fused,
                    task,
                    X_train,
                    y_train,
                    epochs=POST_FUSION_FINETUNE_EPOCHS,
                    seed=seed + stage * 1000 + pair_number,
                )

            vm = metrics_from_predictions(task, y_val, predict_keras(fused, X_val))
            s = fusion_score(task, vm, score_name)
            new_name = f"({a['name']}+{b['name']})"
            history.append(
                {
                    "stage": stage,
                    "pair": pair_number,
                    "model_a": a["name"],
                    "model_b": b["name"],
                    "score_a_before": a["score"],
                    "score_b_before": b["score"],
                    "alpha_reference": alpha_ref,
                    "alpha_other": alpha_other,
                    "reference_swapped_to_higher_score": swapped,
                    "new_model": new_name,
                    "new_score": s,
                    "pairing": pairing,
                    "alignment": alignment,
                    "weighting": weighting,
                    "score_function": score_name,
                    "post_fusion_finetune_epochs": POST_FUSION_FINETUNE_EPOCHS,
                    "post_fusion_finetune_seconds": finetune_seconds,
                }
            )
            new_pool.append(
                {
                    "name": new_name,
                    "model": fused,
                    "val_metrics": vm,
                    "score": s,
                }
            )

        for idx in carries:
            new_pool.append(pool[idx])

        pool = new_pool

    return pool[0]["model"], history


def greedy_model_soup(
    models: Sequence[tf.keras.Model],
    val_metrics: Sequence[Dict[str, float]],
    task: str,
    input_dim: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    score_name: str,
    alignment: bool = True,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    scores = np.asarray([fusion_score(task, m, score_name) for m in val_metrics])
    order = list(np.argsort(-scores))
    ref_idx = order[0]
    ref_weights = models[ref_idx].get_weights()

    prepared: Dict[int, List[np.ndarray]] = {}
    for idx, model in enumerate(models):
        w = model.get_weights()
        if alignment and idx != ref_idx:
            w = align_weight_list_to_reference(ref_weights, w)
        prepared[idx] = [np.array(x, copy=True) for x in w]

    selected = [ref_idx]
    current_weights = prepared[ref_idx]
    current_model = clone_model_with_weights(task, input_dim, current_weights)
    current_metrics = metrics_from_predictions(task, y_val, predict_keras(current_model, X_val))
    current_score = fusion_score(task, current_metrics, score_name)

    for idx in order[1:]:
        candidate_indices = selected + [idx]
        candidate_weights = average_weight_lists(
            [prepared[k] for k in candidate_indices],
            [1.0] * len(candidate_indices),
        )
        candidate_model = clone_model_with_weights(task, input_dim, candidate_weights)
        cm = metrics_from_predictions(task, y_val, predict_keras(candidate_model, X_val))
        cs = fusion_score(task, cm, score_name)
        if cs > current_score:
            selected = candidate_indices
            current_score = cs
            del current_model
            current_model = candidate_model
            current_weights = candidate_weights
        else:
            del candidate_model

    meta = {
        "selected_indices": selected,
        "n_selected": len(selected),
        "reference_index": ref_idx,
        "alignment": alignment,
        "score_function": score_name,
        "validation_score": current_score,
    }
    return current_model, meta


# =============================================================================
# 7. SWA AND FEDAVG BASELINES
# =============================================================================


class SWACollector(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch: int):
        super().__init__()
        self.start_epoch = start_epoch
        self.snapshots: List[List[np.ndarray]] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (epoch + 1) >= self.start_epoch:
            self.snapshots.append([np.array(w, copy=True) for w in self.model.get_weights()])

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.snapshots:
            avg = average_weight_lists(self.snapshots, [1.0] * len(self.snapshots))
            self.model.set_weights(avg)


def train_swa_baseline(
    task: str,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[tf.keras.Model, float]:
    set_global_seed(seed)
    model = create_model(task, input_dim)
    start_epoch = max(1, int(math.ceil(EPOCHS * SWA_START_FRACTION)))
    callback = SWACollector(start_epoch=start_epoch)
    progress("SWA training starting")
    model, elapsed = train_model(
        model, task, X_train, y_train, EPOCHS, seed,
        callbacks=[callback], progress_label="SWA"
    )
    progress(f"SWA training completed in {elapsed:.1f}s")
    return model, elapsed


def make_federated_shards(y: np.ndarray, task: str, n_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_clients = max(1, min(n_clients, len(y)))

    if task == "collision":
        shards: List[List[int]] = [[] for _ in range(n_clients)]
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            for i, sample_idx in enumerate(idx):
                shards[i % n_clients].append(int(sample_idx))
        return [np.asarray(s, dtype=int) for s in shards if len(s) > 0]

    idx = rng.permutation(len(y))
    return [np.asarray(x, dtype=int) for x in np.array_split(idx, n_clients) if len(x) > 0]


def train_fedavg_baseline(
    task: str,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[tf.keras.Model, float]:
    set_global_seed(seed)
    global_model = create_model(task, input_dim)
    global_weights = [np.array(w, copy=True) for w in global_model.get_weights()]
    shards = make_federated_shards(y_train, task, FEDAVG_CLIENTS, seed)

    progress(
        f"FedAvg starting: {FEDAVG_ROUNDS} rounds x {len(shards)} clients "
        f"x {FEDAVG_LOCAL_EPOCHS} local epochs"
    )
    t0 = time.perf_counter()
    for rnd in range(FEDAVG_ROUNDS):
        progress(f"FedAvg round {rnd + 1}/{FEDAVG_ROUNDS} starting")
        local_weight_lists: List[List[np.ndarray]] = []
        local_sizes: List[int] = []
        for client_id, idx in enumerate(shards):
            client_seed = seed + rnd * 1000 + client_id
            set_global_seed(client_seed)
            local_model = create_model(task, input_dim)
            local_model.set_weights([np.array(w, copy=True) for w in global_weights])
            local_model, _ = train_model(
                local_model,
                task,
                X_train[idx],
                y_train[idx],
                epochs=FEDAVG_LOCAL_EPOCHS,
                seed=client_seed,
            )
            local_weight_lists.append([np.array(w, copy=True) for w in local_model.get_weights()])
            local_sizes.append(len(idx))
            del local_model
            progress(
                f"FedAvg round {rnd + 1}/{FEDAVG_ROUNDS} | "
                f"client {client_id + 1}/{len(shards)} completed"
            )
        global_weights = average_weight_lists(local_weight_lists, local_sizes)
        progress(f"FedAvg round {rnd + 1}/{FEDAVG_ROUNDS} aggregation completed")

    global_model.set_weights(global_weights)
    elapsed = time.perf_counter() - t0
    progress(f"FedAvg completed in {elapsed:.1f}s")
    return global_model, elapsed


# =============================================================================
# 8. CLASSICAL BASELINES
# =============================================================================


def make_bagging_classifier(seed: int) -> BaggingClassifier:
    base = LogisticRegression(max_iter=2000)
    try:
        return BaggingClassifier(
            estimator=base,
            n_estimators=NUM_NETWORKS,
            random_state=seed,
            n_jobs=-1,
        )
    except TypeError:
        return BaggingClassifier(
            base_estimator=base,
            n_estimators=NUM_NETWORKS,
            random_state=seed,
            n_jobs=-1,
        )


def make_bagging_regressor(seed: int) -> BaggingRegressor:
    base = LinearRegression()
    try:
        return BaggingRegressor(
            estimator=base,
            n_estimators=NUM_NETWORKS,
            random_state=seed,
            n_jobs=-1,
        )
    except TypeError:
        return BaggingRegressor(
            base_estimator=base,
            n_estimators=NUM_NETWORKS,
            random_state=seed,
            n_jobs=-1,
        )


def build_sklearn_baselines(task: str, seed: int) -> Dict[str, Any]:
    if task == "collision":
        rf = RandomForestClassifier(
            n_estimators=SKLEARN_N_ESTIMATORS,
            random_state=seed,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=SKLEARN_N_ESTIMATORS,
            random_state=seed,
        )
        stacking = StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=SKLEARN_N_ESTIMATORS, random_state=seed, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=SKLEARN_N_ESTIMATORS, random_state=seed)),
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            cv=5,
            n_jobs=-1,
        )
        return {
            "ClassicalBagging_Logistic": make_bagging_classifier(seed),
            "RandomForest": rf,
            "GradientBoosting": gb,
            "Stacking": stacking,
        }

    rf_r = RandomForestRegressor(
        n_estimators=SKLEARN_N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1,
    )
    gb_r = GradientBoostingRegressor(
        n_estimators=SKLEARN_N_ESTIMATORS,
        random_state=seed,
    )
    stacking_r = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=SKLEARN_N_ESTIMATORS, random_state=seed, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=SKLEARN_N_ESTIMATORS, random_state=seed)),
        ],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1,
    )
    return {
        "ClassicalBagging_Linear": make_bagging_regressor(seed),
        "RandomForest": rf_r,
        "GradientBoosting": gb_r,
        "Stacking": stacking_r,
    }


def predict_sklearn(model: Any, task: str, X: np.ndarray) -> np.ndarray:
    if task == "collision":
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))[:, 1]
        if hasattr(model, "decision_function"):
            z = np.asarray(model.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-z))
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def evaluate_sklearn_splits(
    model: Any,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    predictions = {
        "train": predict_sklearn(model, task, X_train),
        "val": predict_sklearn(model, task, X_val),
        "test": predict_sklearn(model, task, X_test),
    }
    y_map = {"train": y_train, "val": y_val, "test": y_test}

    if task == "collision":
        threshold = select_classification_threshold(
            y_val, predictions["val"], CLASSIFICATION_THRESHOLD_OBJECTIVE
        )
        result["decision_threshold"] = float(threshold)
        result["threshold_objective"] = CLASSIFICATION_THRESHOLD_OBJECTIVE

        for split_name in ("train", "val", "test"):
            y = y_map[split_name]
            prob = predictions[split_name]
            metrics = classification_metrics(y, prob, threshold=threshold)
            for k, v in metrics.items():
                result[f"{split_name}_{k}"] = float(v)
            fixed = classification_metrics(y, prob, threshold=0.5)
            for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "predicted_positive_rate"):
                result[f"{split_name}_{k}_at_0_5"] = float(fixed[k])

        result["generalization_gap_auc"] = result["train_roc_auc"] - result["test_roc_auc"]
        result["generalization_gap_logloss"] = result["test_log_loss"] - result["train_log_loss"]
        return result

    for split_name in ("train", "val", "test"):
        metrics = regression_metrics(y_map[split_name], predictions[split_name])
        for k, v in metrics.items():
            result[f"{split_name}_{k}"] = float(v)

    result["generalization_gap_r2"] = result["train_r2"] - result["test_r2"]
    result["generalization_gap_rmse"] = result["test_rmse"] - result["train_rmse"]
    return result


# =============================================================================
# 9. COMPUTATIONAL COST / DIVERSITY
# =============================================================================


def keras_model_size_bytes(model: tf.keras.Model) -> int:
    return int(sum(np.asarray(w).nbytes for w in model.get_weights()))


def sklearn_model_size_bytes(model: Any) -> int:
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def keras_inference_ms_per_sample(model: tf.keras.Model, X: np.ndarray) -> float:
    if len(X) == 0:
        return np.nan
    _ = predict_keras(model, X[: min(16, len(X))])
    t0 = time.perf_counter()
    for _ in range(INFERENCE_REPEATS):
        _ = predict_keras(model, X)
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / (INFERENCE_REPEATS * len(X))


def sklearn_inference_ms_per_sample(model: Any, task: str, X: np.ndarray) -> float:
    if len(X) == 0:
        return np.nan
    _ = predict_sklearn(model, task, X[: min(16, len(X))])
    t0 = time.perf_counter()
    for _ in range(INFERENCE_REPEATS):
        _ = predict_sklearn(model, task, X)
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / (INFERENCE_REPEATS * len(X))


def flatten_weights(model: tf.keras.Model) -> np.ndarray:
    return np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in model.get_weights()])


def pairwise_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def diversity_summary(
    models: Sequence[tf.keras.Model],
    task: str,
    input_dim: int,
    X_val: np.ndarray,
) -> Dict[str, float]:
    preds = [predict_keras(m, X_val) for m in models]
    raw_params = [flatten_weights(m) for m in models]

    pred_disagreements: List[float] = []
    pred_correlations: List[float] = []
    param_dist_raw: List[float] = []
    param_dist_aligned: List[float] = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            p_i, p_j = preds[i], preds[j]
            if task == "collision":
                pred_disagreements.append(float(np.mean((p_i >= 0.5) != (p_j >= 0.5))))
            else:
                pred_disagreements.append(float(np.mean(np.abs(p_i - p_j))))

            if np.std(p_i) > EPS and np.std(p_j) > EPS:
                pred_correlations.append(float(np.corrcoef(p_i, p_j)[0, 1]))

            d_raw = np.linalg.norm(raw_params[i] - raw_params[j]) / math.sqrt(len(raw_params[i]))
            param_dist_raw.append(float(d_raw))

            aligned_j = align_weight_list_to_reference(models[i].get_weights(), models[j].get_weights())
            aligned_flat = np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in aligned_j])
            d_aligned = np.linalg.norm(raw_params[i] - aligned_flat) / math.sqrt(len(raw_params[i]))
            param_dist_aligned.append(float(d_aligned))

    label = "mean_pairwise_class_disagreement" if task == "collision" else "mean_pairwise_prediction_abs_difference"
    return {
        label: pairwise_mean(pred_disagreements),
        "mean_pairwise_prediction_correlation": pairwise_mean(pred_correlations),
        "mean_pairwise_parameter_distance_unaligned": pairwise_mean(param_dist_raw),
        "mean_pairwise_parameter_distance_aligned": pairwise_mean(param_dist_aligned),
    }


# =============================================================================
# 10. RESULT RECORDS / ABLATION SUITE
# =============================================================================


def method_record_keras(
    task: str,
    seed: int,
    method: str,
    model: tf.keras.Model,
    processed: Dict[str, Any],
    fit_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = evaluate_keras_splits(
        model,
        task,
        processed["X_train"],
        processed["y_train"],
        processed["X_val"],
        processed["y_val"],
        processed["X_test"],
        processed["y_test"],
    )
    rec: Dict[str, Any] = {
        "task": task,
        "seed": seed,
        "method": method,
        "fit_seconds": float(fit_seconds),
        "model_params": int(model.count_params()),
        "model_size_bytes": keras_model_size_bytes(model),
        "architecture": ACTIVE_ARCHITECTURE[task],
        "architecture_description": architecture_description(task),
        "deployable_models": 1,
    }
    if RUN_COMPUTE_BENCHMARKS:
        rec["inference_ms_per_sample"] = keras_inference_ms_per_sample(model, processed["X_test"])
    rec.update(metrics)
    if extra:
        rec.update(extra)
    return rec


def method_record_sklearn(
    task: str,
    seed: int,
    method: str,
    model: Any,
    processed: Dict[str, Any],
    fit_seconds: float,
) -> Dict[str, Any]:
    metrics = evaluate_sklearn_splits(
        model,
        task,
        processed["X_train"],
        processed["y_train"],
        processed["X_val"],
        processed["y_val"],
        processed["X_test"],
        processed["y_test"],
    )
    rec: Dict[str, Any] = {
        "task": task,
        "seed": seed,
        "method": method,
        "fit_seconds": float(fit_seconds),
        "model_params": np.nan,
        "model_size_bytes": sklearn_model_size_bytes(model),
        "architecture": "not_applicable",
        "architecture_description": "classical_non_neural_baseline",
        "deployable_models": 1,
    }
    if RUN_COMPUTE_BENCHMARKS:
        rec["inference_ms_per_sample"] = sklearn_inference_ms_per_sample(model, task, processed["X_test"])
    rec.update(metrics)
    return rec


def run_ablation_suite(
    task: str,
    seed: int,
    base_models: Sequence[tf.keras.Model],
    processed: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not RUN_ABLATIONS:
        return []

    input_dim = processed["X_train"].shape[1]
    X_train, y_train = processed["X_train"], processed["y_train"]
    X_val, y_val = processed["X_val"], processed["y_val"]
    X_test, y_test = processed["X_test"], processed["y_test"]
    default_score = DEFAULT_SCORE_FUNCTION[task]

    records: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    def add_recursive(
        label: str,
        n_models: int,
        score_name: str,
        pairing: str,
        alignment: bool,
        weighting: str,
    ) -> None:
        key = ("recursive", n_models, score_name, pairing, alignment, weighting)
        if key in seen:
            return
        seen.add(key)
        progress(
            f"Ablation starting: {label} | recursive | n={n_models} | "
            f"score={score_name} | pairing={pairing} | aligned={alignment} | weighting={weighting}"
        )
        subset = list(base_models[:n_models])
        t0 = time.perf_counter()
        model, _ = recursive_intergen(
            subset,
            task,
            input_dim,
            X_train,
            y_train,
            X_val,
            y_val,
            score_name,
            pairing,
            alignment,
            weighting,
            seed + 991,
            names=[f"ANN{i+1:02d}" for i in range(n_models)],
        )
        elapsed = time.perf_counter() - t0
        metrics = evaluate_keras_splits(model, task, X_train, y_train, X_val, y_val, X_test, y_test)
        rec = {
            "task": task,
            "seed": seed,
            "ablation": label,
            "fusion_mode": "recursive",
            "n_models": n_models,
            "score_function": score_name,
            "pairing": pairing,
            "alignment": alignment,
            "weighting": weighting,
            "construction_seconds": elapsed,
            **metrics,
        }
        records.append(rec)
        progress(f"Ablation completed: {label} in {elapsed:.1f}s")
        del model

    def add_one_step(
        label: str,
        n_models: int,
        score_name: str,
        alignment: bool,
        weighting: str,
    ) -> None:
        key = ("one_step", n_models, score_name, alignment, weighting)
        if key in seen:
            return
        seen.add(key)
        progress(
            f"Ablation starting: {label} | one-step | n={n_models} | "
            f"score={score_name} | aligned={alignment} | weighting={weighting}"
        )
        subset = list(base_models[:n_models])
        val_metrics = validation_metrics_for_models(subset, task, X_val, y_val)
        t0 = time.perf_counter()
        model, _ = one_step_fusion(
            subset,
            val_metrics,
            task,
            input_dim,
            score_name,
            alignment,
            weighting,
        )
        elapsed = time.perf_counter() - t0
        metrics = evaluate_keras_splits(model, task, X_train, y_train, X_val, y_val, X_test, y_test)
        records.append(
            {
                "task": task,
                "seed": seed,
                "ablation": label,
                "fusion_mode": "one_step",
                "n_models": n_models,
                "score_function": score_name,
                "pairing": "not_applicable",
                "alignment": alignment,
                "weighting": weighting,
                "construction_seconds": elapsed,
                **metrics,
            }
        )
        progress(f"Ablation completed: {label} in {elapsed:.1f}s")
        del model

    # A. Recursive vs one-step.
    add_recursive("fusion_mode_recursive", NUM_NETWORKS, default_score, DEFAULT_PAIRING, True, "performance")
    add_one_step("fusion_mode_one_step", NUM_NETWORKS, default_score, True, "performance")

    # B. Performance-guided vs equal coefficients.
    add_recursive("weighting_performance", NUM_NETWORKS, default_score, DEFAULT_PAIRING, True, "performance")
    add_recursive("weighting_equal", NUM_NETWORKS, default_score, DEFAULT_PAIRING, True, "equal")

    # C. Pairing/fusion-order strategies explicitly requested by Reviewer 1.
    for pairing in PAIRING_STRATEGIES:
        add_recursive(f"pairing_{pairing}", NUM_NETWORKS, default_score, pairing, True, "performance")

    # D. Alternative task-appropriate score functions.
    score_functions = CLASSIFICATION_SCORE_FUNCTIONS if task == "collision" else REGRESSION_SCORE_FUNCTIONS
    for score_name in score_functions:
        add_recursive(f"score_{score_name}", NUM_NETWORKS, score_name, DEFAULT_PAIRING, True, "performance")

    # E. Number of networks.
    for n in NETWORK_COUNTS_ABLATION:
        if n <= len(base_models):
            add_recursive(f"n_networks_{n}", n, default_score, DEFAULT_PAIRING, True, "performance")

    # F. Aligned vs unaligned weights.
    add_recursive("alignment_aligned", NUM_NETWORKS, default_score, DEFAULT_PAIRING, True, "performance")
    add_recursive("alignment_unaligned", NUM_NETWORKS, default_score, DEFAULT_PAIRING, False, "performance")

    return records


# =============================================================================
# 11. SUMMARY STATISTICS / WILCOXON TESTS
# =============================================================================


def mean_ci95(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(student_t.ppf(0.975, df=n - 1) * std / math.sqrt(n))
        return mean, std, mean - half, mean + half
    return mean, std, mean, mean


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        c for c in df.columns
        if c.startswith("test_") or c.startswith("generalization_gap_")
        or c in {"fit_seconds", "inference_ms_per_sample", "model_size_bytes"}
    ]
    rows: List[Dict[str, Any]] = []
    for method, group in df.groupby("method"):
        for metric in metric_columns:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue
            mean, std, low, high = mean_ci95(vals)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(vals),
                    "mean": mean,
                    "std": std,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def rank_biserial_from_differences(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > EPS]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    w_pos = float(np.sum(ranks[diff > 0]))
    w_neg = float(np.sum(ranks[diff < 0]))
    denom = w_pos + w_neg
    return 0.0 if denom <= EPS else (w_pos - w_neg) / denom


def paired_statistical_tests(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if not RUN_STATISTICAL_TESTS:
        return pd.DataFrame()

    intergen_name = "INTERGEN_Aligned"
    primary_metric = "test_roc_auc" if task == "collision" else "test_rmse"
    higher_is_better = task == "collision"

    if intergen_name not in set(df["method"]):
        return pd.DataFrame()

    pivot = df.pivot_table(index="seed", columns="method", values=primary_metric, aggfunc="first")
    rows: List[Dict[str, Any]] = []
    raw_p: List[float] = []

    for baseline in pivot.columns:
        if baseline == intergen_name:
            continue
        pair = pivot[[intergen_name, baseline]].dropna()
        if len(pair) < 2:
            continue
        inter = pair[intergen_name].to_numpy(float)
        base = pair[baseline].to_numpy(float)
        # Positive difference always means INTERGEN is better.
        diff = inter - base if higher_is_better else base - inter

        try:
            if np.all(np.abs(diff) <= EPS):
                statistic, p = 0.0, 1.0
            else:
                test = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
                statistic, p = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, p = np.nan, 1.0

        raw_p.append(p)
        rows.append(
            {
                "task": task,
                "primary_metric": primary_metric,
                "comparison": f"{intergen_name} vs {baseline}",
                "n_pairs": len(pair),
                "wilcoxon_statistic": statistic,
                "p_value": p,
                "median_improvement_intergen_positive": float(np.median(diff)),
                "rank_biserial_effect_intergen_positive": rank_biserial_from_differences(diff),
            }
        )

    adjusted = holm_adjust(raw_p)
    for row, adj in zip(rows, adjusted):
        row["p_holm"] = float(adj)
        row["significant_0_05_holm"] = bool(adj < 0.05)
    return pd.DataFrame(rows)


def summarize_ablation(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df
    primary = "test_roc_auc" if task == "collision" else "test_rmse"
    rows: List[Dict[str, Any]] = []
    for ablation, group in df.groupby("ablation"):
        vals = pd.to_numeric(group[primary], errors="coerce").dropna().to_numpy()
        mean, std, low, high = mean_ci95(vals)
        first = group.iloc[0]
        rows.append(
            {
                "ablation": ablation,
                "primary_metric": primary,
                "n": len(vals),
                "mean": mean,
                "std": std,
                "ci95_low": low,
                "ci95_high": high,
                "fusion_mode": first.get("fusion_mode"),
                "n_models": first.get("n_models"),
                "score_function": first.get("score_function"),
                "pairing": first.get("pairing"),
                "alignment": first.get("alignment"),
                "weighting": first.get("weighting"),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 12. MAIN EXPERIMENT PER TASK
# =============================================================================


def run_task(task: str) -> None:
    print("\n" + "=" * 90)
    print(f"Running reviewer-revision experiments for: {task.upper()}")
    print("=" * 90)

    paths = ensure_output_dirs(task)
    task_data = load_task_dataframe(task)
    print(f"Loaded {len(task_data.df):,} rows. {task_data.target_note}")

    main_records: List[Dict[str, Any]] = []
    base_ann_records: List[Dict[str, Any]] = []
    ablation_records: List[Dict[str, Any]] = []
    fusion_history_records: List[Dict[str, Any]] = []
    diversity_records: List[Dict[str, Any]] = []
    alignment_records: List[Dict[str, Any]] = []
    split_summary_records: List[Dict[str, Any]] = []

    for repeat_index, seed in enumerate(SEEDS, start=1):
        seed_started = time.perf_counter()
        print(f"[{task}] repeat {repeat_index}/{len(SEEDS)} | seed={seed}", flush=True)
        progress(f"Seed {seed}: preprocessing and split starting")
        tf.keras.backend.clear_session()
        gc.collect()
        set_global_seed(seed)

        split = split_dataframe(task_data, seed)
        processed = preprocess_split(task_data, split)
        input_dim = processed["X_train"].shape[1]

        split_summary = data_summary(task_data, processed, seed)
        split_summary_records.append(
            {
                "task": task,
                "seed": seed,
                "rows": split_summary["rows_after_target_cleaning"],
                "train_n": split_summary["train_n"],
                "validation_n": split_summary["validation_n"],
                "test_n": split_summary["test_n"],
                "raw_feature_count": split_summary["raw_feature_count"],
                "transformed_feature_count": split_summary["transformed_feature_count"],
                "target_note": split_summary["target_note"],
            }
        )
        dump_json(paths["raw"] / f"data_preprocessing_seed_{seed}.json", split_summary)
        progress(
            f"Seed {seed}: preprocessing complete | train={len(processed['y_train'])}, "
            f"val={len(processed['y_val'])}, test={len(processed['y_test'])}, input_dim={input_dim}"
        )

        # ---------------------------------------------------------------------
        # Base ANNs
        # ---------------------------------------------------------------------
        base_models, base_train_times, model_seeds = train_base_models(
            task,
            input_dim,
            processed["X_train"],
            processed["y_train"],
            seed,
        )
        progress(f"Seed {seed}: evaluating base ANNs on validation set")
        base_val_metrics = validation_metrics_for_models(
            base_models,
            task,
            processed["X_val"],
            processed["y_val"],
        )
        default_score = DEFAULT_SCORE_FUNCTION[task]
        base_scores = [fusion_score(task, m, default_score) for m in base_val_metrics]

        for i, (model, fit_seconds, model_seed, valm, score) in enumerate(
            zip(base_models, base_train_times, model_seeds, base_val_metrics, base_scores), start=1
        ):
            rec = method_record_keras(
                task,
                seed,
                f"ANN{i:02d}",
                model,
                processed,
                fit_seconds,
                extra={
                    "model_seed": model_seed,
                    "fusion_validation_score": score,
                    "fusion_score_function": default_score,
                },
            )
            base_ann_records.append(rec)

        progress(f"Seed {seed}: base ANN metrics recorded")
        # Best individual ANN chosen ONLY on validation score.
        best_idx = int(np.argmax(base_scores))
        progress(f"Seed {seed}: BestSingleANN is ANN{best_idx + 1:02d}")
        main_records.append(
            method_record_keras(
                task,
                seed,
                "BestSingleANN",
                base_models[best_idx],
                processed,
                base_train_times[best_idx],
                extra={"selected_ann_index": best_idx + 1},
            )
        )

        # Fair output-level neural ensemble using exactly the same base ANN pool.
        if RUN_DEEP_ENSEMBLE_ANN:
            progress(f"Seed {seed}: DeepEnsemble_ANN evaluation starting")
            main_records.append(
                method_record_keras_ensemble(
                    task,
                    seed,
                    "DeepEnsemble_ANN",
                    base_models,
                    processed,
                    fit_seconds=sum(base_train_times),
                    extra={
                        "training_data_rule": "all_members_full_training_set",
                        "aggregation": "mean_prediction",
                    },
                )
            )
            progress(f"Seed {seed}: DeepEnsemble_ANN evaluation completed")

        # Fair ANN bagging: same architecture/training budget as INTERGEN, but
        # each member is trained on a bootstrap sample.
        if RUN_ANN_BAGGING:
            progress(f"Seed {seed}: ANN_Bagging architecture-controlled baseline starting")
            ann_bag_models, ann_bag_times = train_ann_bagging(
                task,
                input_dim,
                processed["X_train"],
                processed["y_train"],
                seed + 6500,
                n_models=NUM_NETWORKS,
            )
            main_records.append(
                method_record_keras_ensemble(
                    task,
                    seed,
                    "ANN_Bagging",
                    ann_bag_models,
                    processed,
                    fit_seconds=sum(ann_bag_times),
                    extra={
                        "training_data_rule": "bootstrap_resampling",
                        "aggregation": "mean_prediction",
                        "same_architecture_as_intergen": True,
                    },
                )
            )
            progress(f"Seed {seed}: ANN_Bagging baseline completed")
            del ann_bag_models
            gc.collect()

        # ---------------------------------------------------------------------
        # Permutation-symmetry diagnostics and diversity
        # ---------------------------------------------------------------------
        if len(base_models) >= 2:
            progress(f"Seed {seed}: permutation-alignment diagnostic starting")
            func_error = alignment_function_preservation_error(
                task,
                input_dim,
                base_models[0],
                base_models[1],
                processed["X_val"],
            )
            progress(f"Seed {seed}: alignment diagnostic completed | max prediction change={func_error:.3e}")
            alignment_records.append(
                {
                    "task": task,
                    "seed": seed,
                    "reference": "ANN01",
                    "candidate": "ANN02",
                    "max_abs_prediction_change_after_permutation_alignment": func_error,
                }
            )

        if RUN_DIVERSITY_ANALYSIS:
            progress(f"Seed {seed}: diversity/parameter-distance analysis starting")
            d = diversity_summary(base_models, task, input_dim, processed["X_val"])
            diversity_records.append({"task": task, "seed": seed, **d})
            progress(f"Seed {seed}: diversity analysis completed")

        # ---------------------------------------------------------------------
        # Main INTERGEN (aligned, performance-guided, recursive)
        # ---------------------------------------------------------------------
        progress(f"Seed {seed}: INTERGEN_Aligned construction starting")
        t0 = time.perf_counter()
        intergen_model, history = recursive_intergen(
            base_models,
            task,
            input_dim,
            processed["X_train"],
            processed["y_train"],
            processed["X_val"],
            processed["y_val"],
            default_score,
            DEFAULT_PAIRING,
            DEFAULT_ALIGNMENT,
            "performance",
            seed + 7000,
        )
        intergen_build_seconds = time.perf_counter() - t0
        intergen_rec = method_record_keras(
            task,
            seed,
            "INTERGEN_Aligned",
            intergen_model,
            processed,
            fit_seconds=sum(base_train_times) + intergen_build_seconds,
            extra={
                "pairing": DEFAULT_PAIRING,
                "score_function": default_score,
                "alignment": True,
                "weighting": "performance",
            },
        )
        main_records.append(intergen_rec)
        if task == "collision":
            progress(
                f"Seed {seed}: INTERGEN_Aligned completed in {intergen_build_seconds:.1f}s "
                f"| test AUC={intergen_rec.get('test_roc_auc', float('nan')):.4f} "
                f"| test F1={intergen_rec.get('test_f1', float('nan')):.4f} "
                f"| threshold={intergen_rec.get('decision_threshold', float('nan')):.4f}"
            )
        else:
            progress(
                f"Seed {seed}: INTERGEN_Aligned completed in {intergen_build_seconds:.1f}s "
                f"| test RMSE={intergen_rec.get('test_rmse', float('nan')):.4f} "
                f"| test R2={intergen_rec.get('test_r2', float('nan')):.4f}"
            )
        for h in history:
            fusion_history_records.append({"task": task, "seed": seed, **h})

        if SAVE_FIRST_SEED_INTERGEN_MODEL and repeat_index == 1:
            intergen_model.save(paths["models"] / f"INTERGEN_{task}_seed_{seed}.keras")
            with (paths["models"] / f"preprocessor_{task}_seed_{seed}.pkl").open("wb") as f:
                pickle.dump(processed["preprocessor"], f, protocol=pickle.HIGHEST_PROTOCOL)

        # Explicit unaligned INTERGEN comparison (permutation ablation).
        progress(f"Seed {seed}: INTERGEN_Unaligned construction starting")
        t0 = time.perf_counter()
        intergen_unaligned, _ = recursive_intergen(
            base_models,
            task,
            input_dim,
            processed["X_train"],
            processed["y_train"],
            processed["X_val"],
            processed["y_val"],
            default_score,
            DEFAULT_PAIRING,
            False,
            "performance",
            seed + 7100,
        )
        elapsed_unaligned = time.perf_counter() - t0
        main_records.append(
            method_record_keras(
                task,
                seed,
                "INTERGEN_Unaligned",
                intergen_unaligned,
                processed,
                fit_seconds=sum(base_train_times) + elapsed_unaligned,
            )
        )

        progress(f"Seed {seed}: INTERGEN_Unaligned completed in {elapsed_unaligned:.1f}s")

        # ---------------------------------------------------------------------
        # Reviewer-requested direct weight-fusion baselines
        # ---------------------------------------------------------------------
        if RUN_WEIGHT_BASELINES:
            progress(f"Seed {seed}: direct weight-fusion baselines starting")
            # Naive equal weight average: intentionally unaligned.
            progress("Baseline: EqualWeightAverage_Unaligned")
            t0 = time.perf_counter()
            equal_unaligned, _ = one_step_fusion(
                base_models, base_val_metrics, task, input_dim,
                default_score, alignment=False, weighting="equal"
            )
            main_records.append(
                method_record_keras(
                    task, seed, "EqualWeightAverage_Unaligned", equal_unaligned,
                    processed, sum(base_train_times) + (time.perf_counter() - t0)
                )
            )

            progress("Baseline completed: EqualWeightAverage_Unaligned")
            # Uniform Model Soup with neuron alignment.
            progress("Baseline: ModelSoup_Uniform_Aligned")
            t0 = time.perf_counter()
            uniform_soup, _ = one_step_fusion(
                base_models, base_val_metrics, task, input_dim,
                default_score, alignment=True, weighting="equal"
            )
            main_records.append(
                method_record_keras(
                    task, seed, "ModelSoup_Uniform_Aligned", uniform_soup,
                    processed, sum(base_train_times) + (time.perf_counter() - t0)
                )
            )

            progress("Baseline completed: ModelSoup_Uniform_Aligned")
            # Greedy aligned Model Soup.
            progress("Baseline: ModelSoup_Greedy_Aligned")
            t0 = time.perf_counter()
            greedy_soup, greedy_meta = greedy_model_soup(
                base_models, base_val_metrics, task, input_dim,
                processed["X_val"], processed["y_val"], default_score, alignment=True
            )
            main_records.append(
                method_record_keras(
                    task, seed, "ModelSoup_Greedy_Aligned", greedy_soup,
                    processed, sum(base_train_times) + (time.perf_counter() - t0),
                    extra={"soup_n_selected": greedy_meta["n_selected"]}
                )
            )

            progress("Baseline completed: ModelSoup_Greedy_Aligned")
            # One-step performance-weighted direct fusion.
            progress("Baseline: OneStep_PerformanceWeighted_Aligned")
            t0 = time.perf_counter()
            weighted_one_step, _ = one_step_fusion(
                base_models, base_val_metrics, task, input_dim,
                default_score, alignment=True, weighting="performance"
            )
            main_records.append(
                method_record_keras(
                    task, seed, "OneStep_PerformanceWeighted_Aligned", weighted_one_step,
                    processed, sum(base_train_times) + (time.perf_counter() - t0)
                )
            )

            progress("Baseline completed: OneStep_PerformanceWeighted_Aligned")
            # SWA: same-trajectory checkpoint averaging.
            swa_model, swa_time = train_swa_baseline(
                task, input_dim, processed["X_train"], processed["y_train"], seed + 8000
            )
            main_records.append(
                method_record_keras(task, seed, "SWA", swa_model, processed, swa_time)
            )

            progress("Baseline completed: SWA")
            # FedAvg: data-size weighted client aggregation.
            fedavg_model, fedavg_time = train_fedavg_baseline(
                task, input_dim, processed["X_train"], processed["y_train"], seed + 9000
            )
            main_records.append(
                method_record_keras(task, seed, "FedAvg", fedavg_model, processed, fedavg_time)
            )

        # ---------------------------------------------------------------------
        # Classical ensemble / ML baselines using correct task type
        # ---------------------------------------------------------------------
        if RUN_SKLEARN_BASELINES:
            progress(f"Seed {seed}: classical sklearn baselines starting")
            for name, model in build_sklearn_baselines(task, seed).items():
                progress(f"Sklearn baseline starting: {name}")
                t0 = time.perf_counter()
                model.fit(processed["X_train"], processed["y_train"])
                fit_seconds = time.perf_counter() - t0
                main_records.append(
                    method_record_sklearn(task, seed, name, model, processed, fit_seconds)
                )

        # ---------------------------------------------------------------------
        # Reviewer-requested ablations (one factor at a time)
        # ---------------------------------------------------------------------
        progress(f"Seed {seed}: ablation suite starting")
        ablation_records.extend(run_ablation_suite(task, seed, base_models, processed))
        progress(f"Seed {seed}: ablation suite completed")

        # Release per-seed graph/model memory.
        del base_models, intergen_model, intergen_unaligned
        if RUN_WEIGHT_BASELINES:
            del equal_unaligned, uniform_soup, greedy_soup, weighted_one_step, swa_model, fedavg_model
        tf.keras.backend.clear_session()
        gc.collect()

        # Incremental safety saves in case a long 20-seed run is interrupted.
        pd.DataFrame(main_records).to_csv(paths["raw"] / "main_results_partial.csv", index=False)
        pd.DataFrame(base_ann_records).to_csv(paths["raw"] / "base_ann_metrics_partial.csv", index=False)
        if ablation_records:
            pd.DataFrame(ablation_records).to_csv(paths["raw"] / "ablation_results_partial.csv", index=False)
        progress(f"Seed {seed}: all outputs saved | total seed time={(time.perf_counter() - seed_started) / 60.0:.1f} min")

    # -------------------------------------------------------------------------
    # Final outputs
    # -------------------------------------------------------------------------
    main_df = pd.DataFrame(main_records)
    base_df = pd.DataFrame(base_ann_records)
    ablation_df = pd.DataFrame(ablation_records)
    history_df = pd.DataFrame(fusion_history_records)
    diversity_df = pd.DataFrame(diversity_records)
    alignment_df = pd.DataFrame(alignment_records)
    split_df = pd.DataFrame(split_summary_records)

    main_df.to_csv(paths["raw"] / "main_results.csv", index=False)
    base_df.to_csv(paths["raw"] / "base_ann_metrics.csv", index=False)
    split_df.to_csv(paths["raw"] / "split_summary.csv", index=False)
    if not ablation_df.empty:
        ablation_df.to_csv(paths["raw"] / "ablation_results.csv", index=False)
    if not history_df.empty:
        history_df.to_csv(paths["raw"] / "fusion_history_coefficients.csv", index=False)
    if not diversity_df.empty:
        diversity_df.to_csv(paths["raw"] / "diversity_parameter_distance.csv", index=False)
    if not alignment_df.empty:
        alignment_df.to_csv(paths["raw"] / "permutation_alignment_diagnostic.csv", index=False)

    summary_df = summarize_results(main_df)
    summary_df.to_csv(paths["summary"] / "main_results_mean_ci95.csv", index=False)

    stat_df = paired_statistical_tests(main_df, task)
    if not stat_df.empty:
        stat_df.to_csv(paths["summary"] / "wilcoxon_paired_tests_holm_effect_size.csv", index=False)

    ablation_summary = summarize_ablation(ablation_df, task)
    if not ablation_summary.empty:
        ablation_summary.to_csv(paths["summary"] / "ablation_mean_ci95.csv", index=False)

    # Remove partial files once the full run finishes successfully.
    for partial in paths["raw"].glob("*_partial.csv"):
        try:
            partial.unlink()
        except OSError:
            pass

    print(f"[{task}] Complete. Results saved under: {paths['base'].resolve()}")




# =============================================================================
# 12A. FAST VALIDATION-ONLY ARCHITECTURE SCREENING (basit=True)
# =============================================================================


def screening_record_keras(
    task: str,
    seed: int,
    method: str,
    model: tf.keras.Model,
    processed: Dict[str, Any],
    fit_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validation-only record. The test set is deliberately never evaluated."""
    pred = predict_keras(model, processed["X_val"])
    metrics = metrics_from_predictions(task, processed["y_val"], pred)
    rec: Dict[str, Any] = {
        "task": task,
        "seed": seed,
        "method": method,
        "fit_seconds": float(fit_seconds),
        "model_params": int(model.count_params()),
        "model_size_bytes": keras_model_size_bytes(model),
        "architecture": ACTIVE_ARCHITECTURE[task],
        "architecture_description": architecture_description(task),
        "deployable_models": 1,
        "test_set_evaluated": False,
    }
    rec.update({f"val_{k}": float(v) for k, v in metrics.items()})
    if extra:
        rec.update(extra)
    return rec


def screening_record_ensemble(
    task: str,
    seed: int,
    method: str,
    models: Sequence[tf.keras.Model],
    processed: Dict[str, Any],
    fit_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validation-only mean-prediction ensemble record."""
    pred = _ensemble_prediction(models, processed["X_val"])
    metrics = metrics_from_predictions(task, processed["y_val"], pred)
    rec: Dict[str, Any] = {
        "task": task,
        "seed": seed,
        "method": method,
        "fit_seconds": float(fit_seconds),
        "model_params": int(sum(m.count_params() for m in models)),
        "model_size_bytes": int(sum(keras_model_size_bytes(m) for m in models)),
        "architecture": ACTIVE_ARCHITECTURE[task],
        "architecture_description": architecture_description(task),
        "ensemble_members": len(models),
        "deployable_models": len(models),
        "test_set_evaluated": False,
    }
    rec.update({f"val_{k}": float(v) for k, v in metrics.items()})
    if extra:
        rec.update(extra)
    return rec


def screening_record_sklearn(
    task: str,
    seed: int,
    method: str,
    model: Any,
    processed: Dict[str, Any],
    fit_seconds: float,
) -> Dict[str, Any]:
    """Validation-only classical reference record."""
    pred = predict_sklearn(model, task, processed["X_val"])
    metrics = metrics_from_predictions(task, processed["y_val"], pred)
    rec: Dict[str, Any] = {
        "task": task,
        "seed": seed,
        "method": method,
        "fit_seconds": float(fit_seconds),
        "model_params": np.nan,
        "model_size_bytes": sklearn_model_size_bytes(model),
        "architecture": "not_applicable",
        "architecture_description": "classical_non_neural_baseline",
        "deployable_models": 1,
        "test_set_evaluated": False,
    }
    rec.update({f"val_{k}": float(v) for k, v in metrics.items()})
    return rec


def _screening_summary(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df
    val_metric = "val_roc_auc" if task == "collision" else "val_rmse"
    rows: List[Dict[str, Any]] = []
    for (architecture, method), group in df.groupby(
        ["architecture", "method"], dropna=False
    ):
        vals = pd.to_numeric(
            group[val_metric], errors="coerce"
        ).dropna().to_numpy()
        vmean, vstd, vlow, vhigh = mean_ci95(vals)
        rows.append(
            {
                "task": task,
                "architecture": architecture,
                "method": method,
                "validation_metric": val_metric,
                "validation_n": len(vals),
                "validation_mean": vmean,
                "validation_std": vstd,
                "validation_ci95_low": vlow,
                "validation_ci95_high": vhigh,
                "test_set_evaluated": False,
            }
        )
    return pd.DataFrame(rows)


def _architecture_ranking_from_base_ann(
    base_df: pd.DataFrame,
    method_df: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """
    Recommend architecture from BASE-ANN validation performance only.

    The locked test set is not evaluated in basit=True mode, which avoids
    architecture-selection leakage before the final 20-seed experiment.
    """
    if base_df.empty:
        return pd.DataFrame()

    val_metric = "val_roc_auc" if task == "collision" else "val_rmse"
    higher_better = task == "collision"

    rows: List[Dict[str, Any]] = []
    for architecture, group in base_df.groupby("architecture"):
        vals = pd.to_numeric(
            group[val_metric], errors="coerce"
        ).dropna().to_numpy()
        if len(vals) == 0:
            continue

        inter = method_df[
            (method_df["architecture"] == architecture)
            & (method_df["method"] == "INTERGEN_Aligned")
        ]
        inter_val = pd.to_numeric(
            inter.get(val_metric, pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "task": task,
                "architecture": architecture,
                "selection_basis": f"mean_base_ANN_{val_metric}",
                "mean_base_validation": float(np.mean(vals)),
                "std_base_validation": (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ),
                "n_base_validation_values": int(len(vals)),
                "intergen_validation_mean_diagnostic": (
                    float(inter_val.mean()) if len(inter_val) else np.nan
                ),
                "test_set_evaluated": False,
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    ranking = ranking.sort_values(
        "mean_base_validation",
        ascending=not higher_better,
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "validation_rank", np.arange(1, len(ranking) + 1))
    ranking["recommended_for_final"] = ranking["validation_rank"] == 1
    return ranking


def run_architecture_screening(task: str) -> None:
    """
    Fast pre-screen. This does NOT replace the 20-seed reviewer experiment.

    Scientific guardrails:
    - same split for every architecture within each seed,
    - same optimizer/loss/epoch budget for all ANN methods,
    - same architecture for Single ANN, INTERGEN, Model Soup, SWA, Deep
      Ensemble and ANN Bagging,
    - ANN Bagging differs only by bootstrap training samples,
    - architecture recommendation uses BASE-ANN validation performance only,
    - the locked test set is not evaluated at all in screening mode.
    """
    print("\n" + "=" * 90)
    print(f"FAST ARCHITECTURE SCREENING: {task.upper()} | basit=True")
    print("=" * 90)
    print(
        f"Screening budget: {len(SEEDS)} seeds x {len(SCREEN_ARCHITECTURES[task])} "
        f"architectures x {NUM_NETWORKS} base ANNs x {EPOCHS} epochs"
    )
    print("IMPORTANT: screening results are preliminary, not final manuscript statistics.")

    paths = ensure_output_dirs(task)
    task_data = load_task_dataframe(task)
    print(f"Loaded {len(task_data.df):,} rows. {task_data.target_note}")

    method_records: List[Dict[str, Any]] = []
    base_records: List[Dict[str, Any]] = []
    classical_records: List[Dict[str, Any]] = []

    original_active = ACTIVE_ARCHITECTURE[task]

    for repeat_index, seed in enumerate(SEEDS, start=1):
        progress(f"[SCREEN] {task} seed {repeat_index}/{len(SEEDS)}={seed}: split/preprocess")
        tf.keras.backend.clear_session()
        gc.collect()
        set_global_seed(seed)

        split = split_dataframe(task_data, seed)
        processed = preprocess_split(task_data, split)
        input_dim = processed["X_train"].shape[1]
        default_score = DEFAULT_SCORE_FUNCTION[task]

        # Classical references do not depend on ANN architecture, so train them
        # ONCE per seed rather than repeating them for every architecture.
        if QUICK_RUN_CLASSICAL_REFERENCES:
            progress(f"[SCREEN] {task} seed={seed}: classical references")
            for name, model in build_sklearn_baselines(task, seed).items():
                t0 = time.perf_counter()
                model.fit(processed["X_train"], processed["y_train"])
                rec = screening_record_sklearn(
                    task, seed, name, model, processed, time.perf_counter() - t0
                )
                rec["screening_mode"] = True
                classical_records.append(rec)

        for arch_index, arch_name in enumerate(SCREEN_ARCHITECTURES[task], start=1):
            ACTIVE_ARCHITECTURE[task] = arch_name
            tf.keras.backend.clear_session()
            gc.collect()
            set_global_seed(seed)

            print(
                f"\n[SCREEN] {task} | seed={seed} | architecture "
                f"{arch_index}/{len(SCREEN_ARCHITECTURES[task])}: "
                f"{architecture_description(task)}",
                flush=True,
            )

            base_models, base_times, model_seeds = train_base_models(
                task,
                input_dim,
                processed["X_train"],
                processed["y_train"],
                seed,
            )
            base_val_metrics = validation_metrics_for_models(
                base_models, task, processed["X_val"], processed["y_val"]
            )
            base_scores = [
                fusion_score(task, m, default_score) for m in base_val_metrics
            ]

            for i, (model, fit_seconds, model_seed, valm, score) in enumerate(
                zip(base_models, base_times, model_seeds, base_val_metrics, base_scores),
                start=1,
            ):
                rec = screening_record_keras(
                    task,
                    seed,
                    f"ANN{i:02d}",
                    model,
                    processed,
                    fit_seconds,
                    extra={
                        "model_seed": model_seed,
                        "fusion_validation_score": score,
                        "fusion_score_function": default_score,
                        "screening_mode": True,
                    },
                )
                base_records.append(rec)

            best_idx = int(np.argmax(base_scores))
            method_records.append(
                screening_record_keras(
                    task,
                    seed,
                    "BestSingleANN",
                    base_models[best_idx],
                    processed,
                    base_times[best_idx],
                    extra={
                        "selected_ann_index": best_idx + 1,
                        "screening_mode": True,
                    },
                )
            )

            # Output-level ensemble: same already-trained full-data ANN pool.
            method_records.append(
                screening_record_ensemble(
                    task,
                    seed,
                    "DeepEnsemble_ANN",
                    base_models,
                    processed,
                    fit_seconds=sum(base_times),
                    extra={
                        "training_data_rule": "all_members_full_training_set",
                        "aggregation": "mean_prediction",
                        "screening_mode": True,
                    },
                )
            )

            # Original INTERGEN algorithm, unchanged.
            t0 = time.perf_counter()
            intergen_model, _ = recursive_intergen(
                base_models,
                task,
                input_dim,
                processed["X_train"],
                processed["y_train"],
                processed["X_val"],
                processed["y_val"],
                default_score,
                DEFAULT_PAIRING,
                DEFAULT_ALIGNMENT,
                "performance",
                seed + 7000,
            )
            method_records.append(
                screening_record_keras(
                    task,
                    seed,
                    "INTERGEN_Aligned",
                    intergen_model,
                    processed,
                    fit_seconds=sum(base_times) + (time.perf_counter() - t0),
                    extra={"screening_mode": True},
                )
            )

            # Same base pool -> direct weight-space baselines are architecture
            # controlled automatically.
            t0 = time.perf_counter()
            uniform_soup, _ = one_step_fusion(
                base_models,
                base_val_metrics,
                task,
                input_dim,
                default_score,
                alignment=True,
                weighting="equal",
            )
            method_records.append(
                screening_record_keras(
                    task,
                    seed,
                    "ModelSoup_Uniform_Aligned",
                    uniform_soup,
                    processed,
                    fit_seconds=sum(base_times) + (time.perf_counter() - t0),
                    extra={"screening_mode": True},
                )
            )

            t0 = time.perf_counter()
            greedy_soup, greedy_meta = greedy_model_soup(
                base_models,
                base_val_metrics,
                task,
                input_dim,
                processed["X_val"],
                processed["y_val"],
                default_score,
                alignment=True,
            )
            method_records.append(
                screening_record_keras(
                    task,
                    seed,
                    "ModelSoup_Greedy_Aligned",
                    greedy_soup,
                    processed,
                    fit_seconds=sum(base_times) + (time.perf_counter() - t0),
                    extra={
                        "soup_n_selected": greedy_meta["n_selected"],
                        "screening_mode": True,
                    },
                )
            )

            t0 = time.perf_counter()
            one_step_model, _ = one_step_fusion(
                base_models,
                base_val_metrics,
                task,
                input_dim,
                default_score,
                alignment=True,
                weighting="performance",
            )
            method_records.append(
                screening_record_keras(
                    task,
                    seed,
                    "OneStep_PerformanceWeighted_Aligned",
                    one_step_model,
                    processed,
                    fit_seconds=sum(base_times) + (time.perf_counter() - t0),
                    extra={"screening_mode": True},
                )
            )

            if QUICK_RUN_SWA:
                swa_model, swa_time = train_swa_baseline(
                    task,
                    input_dim,
                    processed["X_train"],
                    processed["y_train"],
                    seed + 8000,
                )
                method_records.append(
                    screening_record_keras(
                        task,
                        seed,
                        "SWA",
                        swa_model,
                        processed,
                        swa_time,
                        extra={"screening_mode": True},
                    )
                )
            else:
                swa_model = None

            if QUICK_RUN_ANN_BAGGING:
                bag_models, bag_times = train_ann_bagging(
                    task,
                    input_dim,
                    processed["X_train"],
                    processed["y_train"],
                    seed + 6500,
                    n_models=NUM_NETWORKS,
                )
                method_records.append(
                    screening_record_ensemble(
                        task,
                        seed,
                        "ANN_Bagging",
                        bag_models,
                        processed,
                        fit_seconds=sum(bag_times),
                        extra={
                            "training_data_rule": "bootstrap_resampling",
                            "aggregation": "mean_prediction",
                            "same_architecture_as_intergen": True,
                            "screening_mode": True,
                        },
                    )
                )
            else:
                bag_models = []

            # Short live comparison for this architecture.
            just = pd.DataFrame(
                [
                    r for r in method_records
                    if r["seed"] == seed and r["architecture"] == arch_name
                ]
            )
            if not just.empty:
                if task == "collision":
                    best_row = just.sort_values("val_roc_auc", ascending=False).iloc[0]
                    inter_val = just.loc[
                        just["method"] == "INTERGEN_Aligned", "val_roc_auc"
                    ].iloc[0]
                    print(
                        f"[SCREEN RESULT] {arch_name}: best neural validation AUC="
                        f"{best_row['val_roc_auc']:.4f} ({best_row['method']}); "
                        f"INTERGEN validation AUC={inter_val:.4f}",
                        flush=True,
                    )
                else:
                    best_row = just.sort_values("val_rmse", ascending=True).iloc[0]
                    inter_val = just.loc[
                        just["method"] == "INTERGEN_Aligned", "val_rmse"
                    ].iloc[0]
                    print(
                        f"[SCREEN RESULT] {arch_name}: best neural validation RMSE="
                        f"{best_row['val_rmse']:.2f} ({best_row['method']}); "
                        f"INTERGEN validation RMSE={inter_val:.2f}",
                        flush=True,
                    )

            # Incremental safety saves.
            pd.DataFrame(method_records).to_csv(
                paths["raw"] / "architecture_screening_methods_partial.csv",
                index=False,
            )
            pd.DataFrame(base_records).to_csv(
                paths["raw"] / "architecture_screening_base_ann_partial.csv",
                index=False,
            )
            if classical_records:
                pd.DataFrame(classical_records).to_csv(
                    paths["raw"] / "architecture_screening_classical_partial.csv",
                    index=False,
                )

            del base_models, intergen_model, uniform_soup, greedy_soup, one_step_model
            if swa_model is not None:
                del swa_model
            for _m in bag_models:
                del _m
            tf.keras.backend.clear_session()
            gc.collect()

    ACTIVE_ARCHITECTURE[task] = original_active

    methods_df = pd.DataFrame(method_records)
    base_df = pd.DataFrame(base_records)
    classical_df = pd.DataFrame(classical_records)

    methods_df.to_csv(
        paths["raw"] / "architecture_screening_methods.csv", index=False
    )
    base_df.to_csv(
        paths["raw"] / "architecture_screening_base_ann.csv", index=False
    )
    if not classical_df.empty:
        classical_df.to_csv(
            paths["raw"] / "architecture_screening_classical_references.csv",
            index=False,
        )

    combined = pd.concat([methods_df, classical_df], ignore_index=True, sort=False)
    summary = _screening_summary(combined, task)
    summary.to_csv(
        paths["summary"] / "architecture_screening_mean_ci95.csv", index=False
    )

    ranking = _architecture_ranking_from_base_ann(base_df, methods_df, task)
    ranking.to_csv(
        paths["summary"] / "architecture_ranking_validation_only.csv", index=False
    )

    if not ranking.empty:
        recommended = str(
            ranking.loc[ranking["recommended_for_final"], "architecture"].iloc[0]
        )
        dump_json(
            paths["summary"] / "recommended_architecture.json",
            {
                "task": task,
                "recommended_architecture": recommended,
                "selection_basis": ranking.iloc[0]["selection_basis"],
                "test_set_used_for_selection": False,
                "screening_repeats": len(SEEDS),
                "screening_networks_per_architecture": NUM_NETWORKS,
                "screening_epochs": EPOCHS,
                "important": (
                    "This is a fast tuning recommendation, not a final statistical "
                    "claim. Set FINAL_ARCHITECTURE for this task to this name and "
                    "rerun with basit=False."
                ),
            },
        )
        print("\n" + "-" * 90)
        print(f"[SCREENING RECOMMENDATION] {task.upper()}: {recommended}")
        print(
            "Selection used BASE-ANN VALIDATION performance only; "
            "the test set was not evaluated at all."
        )
        print(
            f"To run the full reviewer experiment, set "
            f'FINAL_ARCHITECTURE["{task}"] = "{recommended}" and basit = False.'
        )
        print("-" * 90)

    for partial in paths["raw"].glob("architecture_screening_*_partial.csv"):
        try:
            partial.unlink()
        except OSError:
            pass


# =============================================================================
# 12B. PROCESS ISOLATION / RESUMABLE AGGREGATION
# =============================================================================


RAW_RESULT_FILES = (
    "main_results.csv",
    "base_ann_metrics.csv",
    "split_summary.csv",
    "ablation_results.csv",
    "fusion_history_coefficients.csv",
    "diversity_parameter_distance.csv",
    "permutation_alignment_diagnostic.csv",
)


def _worker_seed_output_root(task: str, seed: int) -> Path:
    return OUTPUT_ROOT / "_worker_cache" / task / f"seed_{seed}"


def _worker_raw_dir(task: str, seed: int) -> Path:
    return _worker_seed_output_root(task, seed) / task / "raw"


def _worker_is_complete(task: str, seed: int) -> bool:
    """A cached worker is reusable only when its scientific configuration matches."""
    result_path = _worker_raw_dir(task, seed) / "main_results.csv"
    config_path = _worker_seed_output_root(task, seed) / "configuration.json"
    if not result_path.exists() or not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cached_arch = cfg.get("final_architecture", {}).get(task)
        return (
            cfg.get("mode") == "full_reviewer_experiment"
            and cached_arch == FINAL_ARCHITECTURE[task]
            and int(cfg.get("epochs", -1)) == int(EPOCHS)
            and int(cfg.get("num_networks", -1)) == int(NUM_NETWORKS)
        )
    except Exception:
        return False


def _concat_worker_csvs(task: str, filename: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for seed in SEEDS:
        path = _worker_raw_dir(task, seed) / filename
        if path.exists():
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                print(f"[aggregate] Could not read {path}: {exc}")
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in ("seed", "method", "ablation", "stage", "pair") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return out


def aggregate_isolated_task_outputs(task: str, final: bool) -> None:
    """Merge completed per-seed worker outputs into the normal task directory."""
    paths = ensure_output_dirs(task)

    merged: Dict[str, pd.DataFrame] = {}
    for filename in RAW_RESULT_FILES:
        df = _concat_worker_csvs(task, filename)
        merged[filename] = df
        if df.empty:
            continue

        target_name = filename if final else filename.replace(".csv", "_partial.csv")
        df.to_csv(paths["raw"] / target_name, index=False)

    # Copy per-seed preprocessing metadata and the first deployable model.
    for seed in SEEDS:
        worker_task_root = _worker_seed_output_root(task, seed) / task
        worker_raw = worker_task_root / "raw"
        if worker_raw.exists():
            for js in worker_raw.glob("data_preprocessing_seed_*.json"):
                shutil.copy2(js, paths["raw"] / js.name)

    first_seed = SEEDS[0] if SEEDS else None
    if first_seed is not None:
        worker_models = _worker_seed_output_root(task, first_seed) / task / "models"
        if worker_models.exists():
            for artifact in worker_models.iterdir():
                if artifact.is_file():
                    shutil.copy2(artifact, paths["models"] / artifact.name)

    if not final:
        return

    main_df = merged.get("main_results.csv", pd.DataFrame())
    ablation_df = merged.get("ablation_results.csv", pd.DataFrame())

    if not main_df.empty:
        summarize_results(main_df).to_csv(
            paths["summary"] / "main_results_mean_ci95.csv", index=False
        )
        stat_df = paired_statistical_tests(main_df, task)
        if not stat_df.empty:
            stat_df.to_csv(
                paths["summary"] / "wilcoxon_paired_tests_holm_effect_size.csv",
                index=False,
            )

    if not ablation_df.empty:
        ablation_summary = summarize_ablation(ablation_df, task)
        if not ablation_summary.empty:
            ablation_summary.to_csv(
                paths["summary"] / "ablation_mean_ci95.csv", index=False
            )

    # Canonical full files are now present; remove canonical partial files only.
    for partial in paths["raw"].glob("*_partial.csv"):
        try:
            partial.unlink()
        except OSError:
            pass


def run_task_isolated(task: str) -> None:
    """
    Run every seed in a fresh Python process.

    This is intentionally sequential rather than parallel: it prevents
    TensorFlow memory/thread accumulation without multiplying RAM/GPU pressure,
    and keeps paired-seed experiments reproducible. Completed workers are
    resumable, so an interrupted overnight run does not need to restart seed 1.
    """
    print("\n" + "=" * 90)
    print(f"PROCESS-ISOLATED RUN: {task.upper()} | {len(SEEDS)} seeds")
    print("=" * 90)
    script_path = Path(__file__).resolve()

    completed_durations: List[float] = []
    for repeat_index, seed in enumerate(SEEDS, start=1):
        worker_root = _worker_seed_output_root(task, seed)

        if RESUME_COMPLETED_REPEATS and _worker_is_complete(task, seed):
            print(f"[{task}] seed {repeat_index}/{len(SEEDS)}={seed} already complete -> resume/skip")
            aggregate_isolated_task_outputs(task, final=False)
            continue

        # Remove stale/incompatible cache before starting this seed. This
        # matters when FINAL_ARCHITECTURE is changed after the quick screen.
        if worker_root.exists() and (
            not RESUME_COMPLETED_REPEATS or not _worker_is_complete(task, seed)
        ):
            shutil.rmtree(worker_root, ignore_errors=True)

        env = os.environ.copy()
        env["INTERGEN_WORKER"] = "1"
        env["INTERGEN_SINGLE_TASK"] = task
        env["INTERGEN_SINGLE_SEED"] = str(seed)
        env["INTERGEN_OUTPUT_ROOT"] = str(worker_root.resolve())
        env["INTERGEN_SAVE_MODEL"] = "1" if repeat_index == 1 else "0"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["PYTHONUNBUFFERED"] = "1"

        print(f"[{task}] launching isolated seed {repeat_index}/{len(SEEDS)} | seed={seed}")
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            check=False,
        )
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                f"Worker failed for task={task}, seed={seed}, "
                f"return code={completed.returncode}. Completed earlier seeds remain saved."
            )

        completed_durations.append(elapsed)
        avg_seed = float(np.mean(completed_durations))
        remaining = len(SEEDS) - repeat_index
        eta_hours = (avg_seed * remaining) / 3600.0
        print(
            f"[{task}] isolated seed={seed} completed in {elapsed / 60.0:.1f} min "
            f"| session average={avg_seed / 60.0:.1f} min/seed "
            f"| estimated remaining for {task}={eta_hours:.1f} h",
            flush=True,
        )
        aggregate_isolated_task_outputs(task, final=False)

    aggregate_isolated_task_outputs(task, final=True)
    print(f"[{task}] All isolated seeds aggregated under: {(OUTPUT_ROOT / task).resolve()}")

    if not KEEP_WORKER_OUTPUTS:
        shutil.rmtree(OUTPUT_ROOT / "_worker_cache" / task, ignore_errors=True)


# =============================================================================
# 13. CONFIGURATION SNAPSHOT / ENTRY POINT
# =============================================================================


def configuration_snapshot() -> Dict[str, Any]:
    return {
        "mode": "quick_architecture_screening" if basit else "full_reviewer_experiment",
        "basit": basit,
        "tasks": {"collision": collision, "bike": bike},
        "paths": {"collision": str(COLLISION_FILE), "bike": str(BIKE_FILE)},
        "n_repeats_requested": N_REPEATS,
        "seeds": SEEDS,
        "quick_settings": {
            "repeats": QUICK_REPEATS,
            "num_networks": QUICK_NUM_NETWORKS,
            "epochs": QUICK_EPOCHS,
            "sklearn_n_estimators": QUICK_SKLEARN_N_ESTIMATORS,
            "run_ann_bagging": QUICK_RUN_ANN_BAGGING,
            "run_swa": QUICK_RUN_SWA,
            "run_classical_references": QUICK_RUN_CLASSICAL_REFERENCES,
            "selection_uses_test_set": False,
            "selection_basis": "mean base-ANN validation performance",
        },
        "final_settings": {
            "repeats": FINAL_REPEATS,
            "num_networks": FINAL_NUM_NETWORKS,
            "epochs": FINAL_EPOCHS,
            "sklearn_n_estimators": FINAL_SKLEARN_N_ESTIMATORS,
        },
        "screen_architectures": SCREEN_ARCHITECTURES,
        "final_architecture": FINAL_ARCHITECTURE,
        "active_architecture": ACTIVE_ARCHITECTURE,
        "architecture_library": ARCHITECTURE_LIBRARY,
        "fair_neural_comparison": {
            "same_architecture_within_task": True,
            "same_optimizer": "Adam",
            "same_learning_rate": LEARNING_RATE,
            "same_epoch_budget": EPOCHS,
            "same_batch_size": BATCH_SIZE,
            "ann_bagging_difference": "bootstrap resampling only",
            "deep_ensemble_difference": "output-level mean aggregation only",
        },
        "isolate_repeats_in_subprocess": ISOLATE_REPEATS_IN_SUBPROCESS,
        "resume_completed_repeats": RESUME_COMPLETED_REPEATS,
        "classification_threshold_objective": CLASSIFICATION_THRESHOLD_OBJECTIVE,
        "predict_batch_size": PREDICT_BATCH_SIZE,
        "steps_per_execution": STEPS_PER_EXECUTION,
        "live_progress": LIVE_PROGRESS,
        "progress_every_epochs": PROGRESS_EVERY_EPOCHS,
        "num_networks": NUM_NETWORKS,
        "network_counts_ablation": NETWORK_COUNTS_ABLATION,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "collision_loss": "binary_crossentropy",
        "bike_loss": "mean_squared_error",
        "test_fraction": TEST_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "post_fusion_finetune_epochs": POST_FUSION_FINETUNE_EPOCHS,
        "shared_initialization": SHARED_INITIALIZATION,
        "default_pairing": DEFAULT_PAIRING,
        "default_alignment": DEFAULT_ALIGNMENT,
        "default_score_function": DEFAULT_SCORE_FUNCTION,
        "classification_score_functions": CLASSIFICATION_SCORE_FUNCTIONS,
        "regression_score_functions": REGRESSION_SCORE_FUNCTIONS,
        "pairing_strategies": PAIRING_STRATEGIES,
        "swa_start_fraction": SWA_START_FRACTION,
        "fedavg_clients": FEDAVG_CLIENTS,
        "fedavg_rounds": FEDAVG_ROUNDS,
        "fedavg_local_epochs": FEDAVG_LOCAL_EPOCHS,
        "sklearn_n_estimators": SKLEARN_N_ESTIMATORS,
        "run_weight_baselines": RUN_WEIGHT_BASELINES,
        "run_sklearn_baselines": RUN_SKLEARN_BASELINES,
        "run_deep_ensemble_ann": RUN_DEEP_ENSEMBLE_ANN,
        "run_ann_bagging": RUN_ANN_BAGGING,
        "run_ablations": RUN_ABLATIONS,
        "run_statistical_tests": RUN_STATISTICAL_TESTS,
        "run_diversity_analysis": RUN_DIVERSITY_ANALYSIS,
        "run_compute_benchmarks": RUN_COMPUTE_BENCHMARKS,
        "note_on_larger_datasets": (
            "NYC Taxi / PeMS were requested by reviewers but are not run here because "
            "their data and preprocessing specifications were not supplied."
        ),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    worker_mode = os.environ.get("INTERGEN_WORKER", "0") == "1"
    single_task = os.environ.get("INTERGEN_SINGLE_TASK")

    selected_tasks: List[str] = []
    if single_task:
        if single_task not in {"collision", "bike"}:
            raise ValueError(f"Invalid INTERGEN_SINGLE_TASK={single_task}")
        selected_tasks = [single_task]
    else:
        if collision:
            selected_tasks.append("collision")
        if bike:
            selected_tasks.append("bike")

    if not selected_tasks:
        raise ValueError("At least one of `collision` or `bike` must be True.")

    # Parent snapshot describes the requested complete experiment. Worker
    # snapshots remain inside their own cache directories.
    dump_json(OUTPUT_ROOT / "configuration.json", configuration_snapshot())
    dump_json(OUTPUT_ROOT / "environment.json", environment_info())

    print("Selected tasks:", ", ".join(selected_tasks))
    print(
        "Mode:",
        "BASIT=True -> FAST ARCHITECTURE SCREENING"
        if basit else
        "BASIT=False -> FULL REVIEWER EXPERIMENT",
    )
    print(f"Seeds in this process: {len(SEEDS)}; base ANNs per architecture/seed: {NUM_NETWORKS}")
    print("Main INTERGEN:", {
        "pairing": DEFAULT_PAIRING,
        "alignment": DEFAULT_ALIGNMENT,
        "score": DEFAULT_SCORE_FUNCTION,
        "post_fusion_finetune_epochs": POST_FUSION_FINETUNE_EPOCHS,
        "classification_threshold": (
            f"validation_{CLASSIFICATION_THRESHOLD_OBJECTIVE}"
            if "collision" in selected_tasks else "n/a"
        ),
        "final_architecture": FINAL_ARCHITECTURE,
    })

    if basit:
        # Deliberately compact tuning run. It does not run the reviewer
        # statistical suite and must not be used as the final manuscript table.
        enable_determinism()
        for task in selected_tasks:
            run_architecture_screening(task)
    else:
        # Worker processes and explicitly non-isolated runs execute TensorFlow
        # experiments directly. The parent orchestrator itself avoids doing
        # long-lived model work, preventing graph/thread accumulation.
        if worker_mode or not ISOLATE_REPEATS_IN_SUBPROCESS:
            enable_determinism()
            for task in selected_tasks:
                ACTIVE_ARCHITECTURE[task] = FINAL_ARCHITECTURE[task]
                run_task(task)
        else:
            for task in selected_tasks:
                run_task_isolated(task)

    print("\nAll requested tasks finished.")
    print(f"Output root: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()

