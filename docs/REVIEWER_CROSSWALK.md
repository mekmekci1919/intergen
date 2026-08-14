# Reviewer-request crosswalk

This document maps the major reviewer requests to the implemented code paths.

| Reviewer concern | Repository implementation |
|---|---|
| Permutation symmetry | `align_weight_list_to_reference()` uses Hungarian matching; aligned vs unaligned ablation and function-preservation diagnostic are produced. |
| Model Soup / direct parameter merging | Uniform and greedy aligned Model Soup plus one-step performance-weighted fusion are included. |
| SWA and FedAvg | Both direct baselines are implemented in the main experiment. |
| Train/validation/test separation | 60/20/20 split; preprocessing is fitted on training only. |
| Architecture/hyperparameters | Explicit task architectures, Adam, learning rate, batch size, epoch budget and losses are stored in code and configuration snapshots. |
| Fusion order | Best-worst is the main policy; similar and random pairing are ablations. |
| Alternative scores | AUC/F1/balanced-accuracy/inverse-log-loss for classification; R²/MSE, inverse MSE, inverse MAE and positive R² for regression. |
| More repeated runs | 20 paired seeds by default. |
| Statistical testing | Paired two-sided Wilcoxon, Holm correction, rank-biserial effect size. |
| Classification-native metrics | ROC-AUC, F1, precision, recall, balanced accuracy, accuracy, log loss and Brier. |
| Different network counts | 4/8/16-network ablation. |
| Diversity / parameter distance | Raw diversity, disagreement, alignment-distance and fusion-coefficient outputs. |
| Compute/deployment | Fit/construction time, inference time, model size and parameter counts. |
| Dataset-selection clarification | Supplied collision file is explicitly Lancashire (`police_force=4`, n=2,762); bike is the 731-row daily file. |
| Larger NYC Taxi / PeMS validation | Not fabricated in code; documented as future large-scale spatiotemporal validation, consistent with the revised manuscript. |

The experiment intentionally preserves negative findings. In particular,
alignment is treated as necessary for meaningful direct fusion, and the bike
regression result is not presented as evidence of universal superiority.
