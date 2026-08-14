#!/usr/bin/env python3
"""Lightweight validation of the two INTERGEN repository datasets.

This script intentionally does not import TensorFlow. It checks the exact study
scope and the train/validation/test preprocessing dimensions reported in the
revised manuscript.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COLLISION = DATA / "collision_2023_lancashire.xlsx"
BIKE = DATA / "day_bike_share.xlsx"

SEED = 20260811
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.20

COLLISION_NUMERIC = [
    "number_of_vehicles", "number_of_casualties", "speed_limit"
]
COLLISION_CATEGORICAL = [
    "day_of_week", "first_road_class", "road_type", "junction_control",
    "pedestrian_crossing_human_control",
    "pedestrian_crossing_physical_facilities",
    "weather_conditions", "road_surface_conditions", "accident_severity",
]
BIKE_NUMERIC = ["temp", "atemp", "hum", "windspeed"]
BIKE_CATEGORICAL = [
    "season", "yr", "mnth", "holiday", "weekday", "workingday"
]

def one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def transformed_dimension(df, numeric, categorical, target, stratify):
    X = df[numeric + categorical].copy()
    y = df[target].to_numpy()
    strat = y if stratify else None
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=strat
    )
    rel_val = VALIDATION_FRACTION / (1.0 - TEST_FRACTION)
    strat_dev = y_dev if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=rel_val, random_state=SEED + 17,
        stratify=strat_dev
    )
    pre = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", one_hot()),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )
    transformed = pre.fit_transform(X_train)
    return len(X_train), len(X_val), len(X_test), transformed.shape[1]

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def main():
    require(COLLISION.exists(), f"Missing {COLLISION}")
    require(BIKE.exists(), f"Missing {BIKE}")

    c = pd.read_excel(COLLISION)
    b = pd.read_excel(BIKE)

    require(len(c) == 2762, f"Collision rows: expected 2762, got {len(c)}")
    require("police_force" in c.columns, "Collision data missing police_force")
    pf = pd.to_numeric(c["police_force"], errors="coerce")
    require(pf.notna().all() and (pf == 4).all(),
            "Collision file is not exclusively police_force=4 (Lancashire)")
    require("urban_or_rural_area" in c.columns,
            "Collision target urban_or_rural_area is missing")
    counts = c["urban_or_rural_area"].value_counts(dropna=False).to_dict()
    require(counts.get(1, 0) == 1806 and counts.get(0, 0) == 956,
            f"Collision target distribution mismatch: {counts}")

    require(len(b) == 731, f"Bike rows: expected 731, got {len(b)}")
    require("cnt" in b.columns, "Bike target cnt is missing")
    require("hr" not in b.columns and "hour" not in b.columns,
            "Daily bike file unexpectedly contains an hourly predictor")

    c_dims = transformed_dimension(
        c, COLLISION_NUMERIC, COLLISION_CATEGORICAL,
        "urban_or_rural_area", stratify=True
    )
    b_dims = transformed_dimension(
        b, BIKE_NUMERIC, BIKE_CATEGORICAL, "cnt", stratify=False
    )

    require(c_dims == (1656, 553, 553, 57),
            f"Collision split/features mismatch: {c_dims}")
    require(b_dims == (438, 146, 147, 33),
            f"Bike split/features mismatch: {b_dims}")

    print("INTERGEN data validation: PASS")
    print(f"Collision: n={len(c)}, police_force=4, target 1/0={counts}, "
          f"split/features={c_dims}")
    print(f"Bike: n={len(b)}, split/features={b_dims}")

if __name__ == "__main__":
    main()
