#!/usr/bin/env python3
"""Create the Lancashire collision subset from a full 2023 STATS19 collision file.

Usage:
    python scripts/prepare_lancashire_from_stats19.py path/to/full_collision_file.csv

CSV and XLSX inputs are supported. The output is written to:
    data/collision_2023_lancashire.xlsx

The target is converted from original STATS19 coding
    1 = Urban, 2 = Rural
to the analysis coding
    Urban = 1, Rural = 0.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "collision_2023_lancashire.xlsx"

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input extension: {suffix}")

def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/prepare_lancashire_from_stats19.py "
            "path/to/full_2023_collision_file.csv"
        )

    source = Path(sys.argv[1]).expanduser().resolve()
    df = read_table(source)

    required = {"police_force", "urban_or_rural_area"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    pf = pd.to_numeric(df["police_force"], errors="coerce")
    out = df.loc[pf == 4].copy()

    raw_target = pd.to_numeric(out["urban_or_rural_area"], errors="coerce")
    if set(raw_target.dropna().unique()).issubset({0, 1}):
        # Already in analysis coding.
        out["urban_or_rural_area"] = raw_target
    else:
        out["urban_or_rural_area"] = raw_target.map({1: 1, 2: 0})

    out = out.loc[out["urban_or_rural_area"].notna()].copy()
    out["urban_or_rural_area"] = out["urban_or_rural_area"].astype(int)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUTPUT, index=False)

    counts = out["urban_or_rural_area"].value_counts().to_dict()
    print(f"Wrote: {OUTPUT}")
    print(f"Rows: {len(out)}")
    print(f"police_force values: {sorted(out['police_force'].dropna().unique().tolist())}")
    print(f"Target counts (Urban=1, Rural=0): {counts}")

if __name__ == "__main__":
    main()
