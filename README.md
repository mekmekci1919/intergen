# INTERGEN 

This repository contains the unified Python implementation of the INTERGEN reviewer-revision experiments for two tasks:

- **Collision**: binary classification of `urban_or_rural_area`
- **Bike sharing**: regression of daily bike-share count `cnt`

The script includes strict train/validation/test separation, task-appropriate metrics, aligned and unaligned weight-fusion baselines, Model Soup variants, SWA, FedAvg, ablations, repeated-seed statistics, diversity diagnostics, computational-cost outputs, and a fast validation-only architecture screening mode.

## Repository structure

```text
.
├── intergen_experiment.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── README.md
│   └── .gitkeep
└── .github/
    └── workflows/
        └── syntax-check.yml
```

Generated result directories and local datasets are intentionally excluded from Git.

## 1. Clone and create an environment

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
python -m venv .venv
```

Activate the environment and install dependencies:

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Add the datasets

By default, place these two files in `data/`:

```text
data/collision_2023_clevand.xlsx
data/day_bike_share.xlsx
```

The code no longer depends on a machine-specific `H:\...` path.

If the datasets are stored somewhere else, copy `.env.example` to `.env` and set either the data directory or the individual file paths.

Example:

```dotenv
INTERGEN_DATA_DIR=D:/research/intergen/data
```

or:

```dotenv
INTERGEN_COLLISION_FILE=D:/research/data/collision_2023_clevand.xlsx
INTERGEN_BIKE_FILE=D:/research/data/day_bike_share.xlsx
```

## 3. Fast architecture screening

Use this first when you want the small validation-only screening run. The locked test set is not used for architecture selection.

### Windows PowerShell

```powershell
$env:INTERGEN_BASIT="1"
python intergen_experiment.py
```

### macOS / Linux

```bash
INTERGEN_BASIT=1 python intergen_experiment.py
```

You can also copy `.env.example` to `.env`, leave `INTERGEN_BASIT=1`, and run:

```bash
python intergen_experiment.py
```

Architecture recommendations are written under the screening result directory. After screening, set the chosen names in `FINAL_ARCHITECTURE` before the final experiment if they differ from the currently frozen choices.

## 4. Full reviewer-facing experiment

The default source configuration is the full experiment. It uses the selected final architectures, repeated paired seeds, direct weight-space baselines, ablations, statistical tests, and process isolation/resume logic.

### Windows PowerShell

```powershell
$env:INTERGEN_BASIT="0"
python intergen_experiment.py
```

### macOS / Linux

```bash
INTERGEN_BASIT=0 python intergen_experiment.py
```

The full experiment is computationally intensive. Completed isolated seeds are cached and can be resumed when the scientific configuration matches.

## 5. Run only one task

### Collision only

```dotenv
INTERGEN_RUN_COLLISION=1
INTERGEN_RUN_BIKE=0
```

### Bike only

```dotenv
INTERGEN_RUN_COLLISION=0
INTERGEN_RUN_BIKE=1
```

These switches can be placed in `.env` or supplied as shell environment variables.

## Main outputs

Depending on the mode, the script writes raw and summarized CSV/JSON outputs including:

- main method results
- base ANN metrics
- train/validation/test split summaries
- fusion-history coefficients
- permutation-alignment diagnostics
- diversity and parameter-distance statistics
- ablation results
- 95% confidence-interval summaries
- paired Wilcoxon tests with Holm correction and rank-biserial effect sizes
- saved first-seed INTERGEN model and preprocessing pipeline when enabled
- environment and configuration snapshots

## Reproducibility notes

- Preprocessing is fitted on the training partition only.
- Architecture screening uses validation performance only for selection.
- The full experiment uses paired seeds.
- TensorFlow deterministic operations are requested when supported.
- The script stores configuration and package-version snapshots with the results.
- Classification thresholds are selected on validation data and then frozen for test reporting.

## Data availability

The datasets are not included in this repository by default. Before publishing them, verify that their original licenses or data-use terms allow redistribution. If redistribution is not permitted, keep the files outside Git and document where readers can obtain them.

## Larger external datasets

The current implementation does not fabricate preprocessing for reviewer-requested external datasets such as NYC Taxi or PeMS because those datasets and preprocessing specifications were not supplied. The task-loading framework can be extended later without changing the fusion and statistical-analysis machinery.
