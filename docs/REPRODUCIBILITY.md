# Reproducibility workflow

## 1. Create the environment

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -r requirements.txt
```

## 2. Validate the repository data

```bash
python scripts/validate_data.py
```

This check is intentionally independent of TensorFlow.

## 3. Confirm the experiment configuration

The full reviewer experiment is the default:

```python
basit = False
FINAL_REPEATS = 20
FINAL_NUM_NETWORKS = 16
FINAL_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
POST_FUSION_FINETUNE_EPOCHS = 0
SHARED_INITIALIZATION = False
DEFAULT_PAIRING = "best_worst"
DEFAULT_ALIGNMENT = True
```

Final architectures frozen after validation-only screening:

- Collision: submitted/original 14-hidden-layer architecture, 16 units/layer,
  with the activation sequence stored directly in `intergen_experiment.py`.
- Bike share: five hidden Dense layers, each 32 ReLU units.

## 4. Run the full experiment

```bash
python intergen_experiment.py
```

The default output root is:

```text
results/intergen_revision_final_fair_v7_results/
```

The full run uses per-seed process isolation. Completed seed workers are
resumable.

## 5. Expected raw files

For each task the run aggregates:

- `main_results.csv`
- `base_ann_metrics.csv`
- `split_summary.csv`
- `ablation_results.csv`
- `fusion_history_coefficients.csv`
- `diversity_parameter_distance.csv`
- `permutation_alignment_diagnostic.csv`
- per-seed preprocessing JSON files

## 6. Expected summary files

- `main_results_mean_ci95.csv`
- `wilcoxon_paired_tests_holm_effect_size.csv`
- `ablation_mean_ci95.csv`

The root also contains:

- `configuration.json`
- `environment.json`

## 7. Before publishing generated outputs

Check that:

1. All 20 paired seeds are present.
2. Collision uses only the supplied Lancashire file / `police_force = 4`.
3. Bike uses 731 daily observations.
4. The summary values agree with the final manuscript tables.
5. No `_partial.csv` files are presented as final results.
6. No worker cache or large saved Keras models need to be committed unless
   specifically required.

## Important scientific-integrity rule

The manuscript means and confidence intervals in `reference/` are included
only as a cross-check. They must never be expanded into synthetic per-seed
data. Repository raw results should come from the actual experiment run.
