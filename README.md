# INTERGEN

**INTERGEN: Performance-Guided Recursive Neural Network Weight Fusion for Transportation Prediction**

This repository contains the reviewer-revision experiment, the two analysis
datasets used in the revised manuscript, reproducibility checks, and the code
that generates the raw and summarized experimental outputs.

## Repository status

The repository package is aligned with the revised manuscript:

- Collision task: **2023 STATS19, Lancashire police-force area**
  (`police_force = 4`), **n = 2,762**.
- Collision target used by the experiment: **Urban = 1, Rural = 0**
  (1,806 Urban; 956 Rural).
- Bike-share task: **731 daily Capital Bikeshare observations**.
- Split protocol: **60% train / 20% validation / 20% test** for each of
  20 paired seeds.
- Base ANN count: **16** per seed.
- Main collision fusion score: validation **ROC-AUC**.
- Main bike fusion score: `max(R², 0) / MSE`.
- Permutation handling: layer-wise **Hungarian neuron matching** before
  aligned parameter interpolation.
- Main pairing rule: **best-worst**, with similar/random pairing ablations.
- Statistical analysis: paired two-sided **Wilcoxon signed-rank tests**,
  **Holm correction**, and **rank-biserial effect sizes**.

The obsolete local collision filename and location wording from earlier development
versions are **not used by the experiment in this repository**.

## Files

```text
INTERGEN/
├── intergen_experiment.py
├── requirements.txt
├── README.md
├── SHA256SUMS.txt
├── .gitignore
├── .gitattributes
├── data/
│   ├── collision_2023_lancashire.xlsx
│   ├── day_bike_share.xlsx
│   └── README.md
├── scripts/
│   ├── validate_data.py
│   └── prepare_lancashire_from_stats19.py
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── REVIEWER_CROSSWALK.md
├── reference/
│   ├── README.md
│   └── manuscript_reported_results.csv
├── results/
│   └── README.md
└── .github/
    └── workflows/
        └── validate.yml
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate the supplied datasets first

```bash
python scripts/validate_data.py
```

Expected checks include:

- collision rows = 2,762;
- all collision `police_force` values = 4;
- collision target distribution = 1,806 Urban / 956 Rural;
- collision split = 1,656 / 553 / 553 and 57 transformed inputs;
- bike rows = 731;
- bike split = 438 / 146 / 147 and 33 transformed inputs.

## Run the reviewer-facing experiment

The supplied script is configured with `basit = False`, which is the full
reviewer-facing experiment.

```bash
python intergen_experiment.py
```

By default both tasks are run. The code uses fresh subprocesses for individual
seeds in the full experiment and supports resuming completed seeds.

Run one task only:

Windows PowerShell:

```powershell
$env:INTERGEN_SINGLE_TASK="collision"
python intergen_experiment.py
```

macOS/Linux:

```bash
INTERGEN_SINGLE_TASK=collision python intergen_experiment.py
```

A one-seed reproducibility/smoke run can be requested with
`INTERGEN_SINGLE_SEED=20260811`. This is **not** a substitute for the full
20-seed manuscript analysis.

## Data paths

No author-specific `H:\...` path is required. The default files are:

```text
data/collision_2023_lancashire.xlsx
data/day_bike_share.xlsx
```

Optional environment overrides:

- `INTERGEN_DATA_DIR`
- `INTERGEN_COLLISION_FILE`
- `INTERGEN_BIKE_FILE`
- `INTERGEN_OUTPUT_ROOT`

The collision loader also accepts a full 2023 STATS19 collision file and, when
needed, automatically restricts it to `police_force == 4`. A separate helper
is included for reconstructing the exact analysis subset:

```bash
python scripts/prepare_lancashire_from_stats19.py path/to/full_collision_file.csv
```

## Outputs

The experiment writes a configuration and environment snapshot plus task-level
raw and summary files. Key raw outputs include:

```text
main_results.csv
base_ann_metrics.csv
split_summary.csv
ablation_results.csv
fusion_history_coefficients.csv
diversity_parameter_distance.csv
permutation_alignment_diagnostic.csv
```

Summary outputs include:

```text
main_results_mean_ci95.csv
wilcoxon_paired_tests_holm_effect_size.csv
ablation_mean_ci95.csv
```

The `results/` folder in this package contains a README only. **Do not invent
or reconstruct per-seed raw results from manuscript means.** Run the supplied
experiment and then commit the actual generated CSV/JSON outputs if the
repository is intended to substantiate the manuscript's raw-result
availability statement.

## Reproducibility notes

The final experiment uses:

- 20 paired seeds beginning at `20260811`;
- 16 independently initialized base ANNs;
- 100 epochs;
- batch size 32;
- Adam, learning rate 0.001;
- no post-fusion fine-tuning;
- validation-selected classification threshold;
- classification metrics including ROC-AUC, F1, precision, recall, balanced
  accuracy, log loss and Brier score;
- regression metrics MSE, RMSE, MAE and R².

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete
workflow.

## Data provenance

The collision analysis file is the Lancashire (`police_force = 4`) subset of
the 2023 Great Britain STATS19 collision data. The bike-share analysis uses
the 731-row daily Capital Bikeshare dataset described by Fanaee-T and Gama
(2013). Data remain subject to the terms of their original providers.

## Scientific scope

The revised manuscript deliberately limits empirical claims to these two
tabular transportation tasks. NYC Taxi and PeMS are identified as future
large-scale spatiotemporal validation rather than being represented as part of
the current experiment.
