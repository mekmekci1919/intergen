# Data

This folder contains the two files used by the revised INTERGEN experiment.

## `collision_2023_lancashire.xlsx`

Validated properties:

- 2,762 rows.
- Every row has `police_force = 4` (Lancashire police-force area).
- Analysis target: `urban_or_rural_area`.
- Target coding in this repository file: Urban = 1, Rural = 0.
- Target distribution: 1,806 Urban and 956 Rural.
- No additional collision-level sampling is performed by the experiment.

The repository loader can also accept a full national 2023 STATS19 collision
file and will restrict it to `police_force == 4` before analysis.

## `day_bike_share.xlsx`

Validated properties:

- 731 rows.
- Daily observations only.
- Regression target: `cnt`.
- No `hour`/`hr` predictor is used.

## Reproduce the manuscript preprocessing dimensions

Run:

```bash
python scripts/validate_data.py
```

For seed 20260811 the script verifies the manuscript-reported split sizes and
transformed input dimensions:

- Collision: 1,656 train / 553 validation / 553 test; 57 transformed inputs.
- Bike: 438 train / 146 validation / 147 test; 33 transformed inputs.
