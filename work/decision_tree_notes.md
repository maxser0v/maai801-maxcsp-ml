# Decision Tree validation notes

## Setup

`work/decision_tree.py` implements the shared `train_and_validate(fit_df, val_df, features, target)` interface. It evaluates the declared 12-combination grid of `max_depth` = 3, 5, 8, or unbounded and `min_samples_leaf` = 1, 5, or 10, with `random_state=42`. The selection criterion is validation mean absolute error (MAE). Exact ties prefer smaller finite depth, then larger leaf size; unbounded depth is ranked as the least simple option.

The shared `results/split_manifest.csv` has 128 fit, 32 validation, and 40 supplied-test instance IDs. The runner checks that this saved manifest exactly matches the manifest recreated by `work/data_exploration.py`; it fits only on the 5,120 fit candidate rows and scores only the 1,280 validation rows. The supplied test partition is not used.

## Validation results

| max_depth | min_samples_leaf | Validation MAE |
|---:|---:|---:|
| 3 | 1 | 0.576804 |
| 3 | 5 | 0.576804 |
| 3 | 10 | 0.576804 |
| 5 | 1 | 0.531444 |
| 5 | 5 | 0.531444 |
| 5 | 10 | 0.531780 |
| 8 | 1 | 0.523920 |
| 8 | 5 | 0.522591 |
| 8 | 10 | **0.518270** |
| unbounded | 1 | 0.561947 |
| unbounded | 5 | 0.531224 |
| unbounded | 10 | 0.521205 |

Selected parameters are `max_depth=8`, `min_samples_leaf=10`, and `random_state=42`, with validation MAE **0.518270**. The model is returned fitted on the fit partition. `results/dt_validation.csv` is the machine-readable record for all combinations, including the selected-row marker.

## Interpretation

Validation MAE generally falls as the permitted depth grows from 3 to 8, suggesting the shallowest trees underfit this feature-to-improvement mapping relative to the selected tree. The unbounded trees score worse than their depth-8 counterparts, consistent with additional complexity fitting training detail that does not carry as well to held-out instances. These are validation-based observations only; final test evaluation and selection between model families remain with the coordinator.

## Sources and reproduction

- `supplied/MAAI801_MaxCSP_Student_Package/student_starter.ipynb`, cells 7–8: Decision Tree task, grouped `GroupShuffleSplit` recipe, and sklearn fit/MAE pattern.
- `work/data_exploration.py`: shared six-feature/target definitions and grouped partition loader.
- `supplied/MAAI801_MaxCSP_Student_Package/maxcsp_tools.py`, lines 48–63: pre-mutation feature definitions and best-value mutation target semantics.
- Run from the project root with `python3 work/decision_tree.py` to verify the saved manifest, rerun grid tuning, and write `results/dt_validation.csv`.
