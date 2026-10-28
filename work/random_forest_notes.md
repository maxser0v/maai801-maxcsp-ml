# Random Forest validation notes

## Method

The Random Forest uses the shared `fit_df` and `val_df` produced by `work/data_exploration.py`, with the six agreed pre-mutation features and `improvement` target. The split groups by `instance_id`; the model grid has eight configurations: `n_estimators` in `{100, 200}`, `max_depth` in `{5, None}`, and `min_samples_leaf` in `{1, 5}`. Every run sets `random_state=42`; validation selection uses mean absolute error (MAE). Exact MAE ties are resolved in order by fewer trees, shallower depth, then larger leaves.

## Result

The selected configuration is `n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`, `random_state=42`. Its shared validation MAE is **0.514774** violations per candidate. All eight settings and scores are in `results/rf_validation.csv`; the selected settings are also in `results/rf_best_params.json`.

The selected forest's fit MAE is 0.360763, below its validation MAE of 0.514774. This gap indicates that the forest fits the training candidates more closely than it generalizes to unseen instances, even with five samples required per leaf. These metrics measure prediction error only; they do not establish whether the model makes better one-step variable choices than the heuristic.

## Reproduction

From the project root, run:

```sh
python3 work/random_forest.py
```

The script loads the shared split and writes the validation table and selected-parameter JSON. It does not evaluate or tune on test outcomes.

## Source references

- `ASSIGNMENT_PLAN.md`, Phase 3 (grid, grouped split, tie rule, and test-data guard).
- `supplied/MAAI801_MaxCSP_Student_Package/student_starter.ipynb`, cells 9–10 (Random Forest task and required estimator).
- `work/data_exploration.py`, `FEATURES`, `TARGET`, and `load_data` (shared feature and grouped-split contract).
