# Optional search across additional models

Read [RESULTS.md](RESULTS.md) for the experiment and findings. We tried 45 configurations across eight model families, using only the original training instances for model selection.

A standardized Support Vector Regressor won both selection criteria: lowest cross-validation MAE and highest cross-validation actual selection gain. It achieved 7.03% above the heuristic on the previously evaluated original test sample, below the original forest's 7.99%. On 2,500 fresh synthetic states, it achieved 3.31%, compared with 1.76% for the forest.

## Reproduce

From the repository root, with the dependencies in `submission/requirements.txt` installed:

```bash
python3 experiments/model_search/search.py
python3 experiments/model_search/evaluate.py
python3 experiments/model_search/summarize.py
```

The search uses four-fold validation grouped by problem instance. It saves its rules before fitting, fixes the chosen configurations before evaluation, and refits those configurations on all original training instances. The fresh evaluation follows the supplied generator's distribution and uses two recorded seed streams.

## Files

| Files | Contents |
|---|---|
| `search.py`, `evaluate.py`, `summarize.py` | Search, frozen-model evaluation, and report/figure generation |
| `RESULTS.md`, `report_text.md` | Readable report and its editorial source |
| `protocol.json`, `selection_lock.json` | Search rules and selected models |
| `cv_instance_folds.csv` | Instance membership in each validation fold |
| `cv_leaderboard.csv`, `cv_fold_metrics.csv`, `best_per_family_cv.csv` | Complete search scores and family summaries |
| `trial_*_oof.npy` | Out-of-fold predictions for each configuration |
| `*.joblib` | Fitted original forest and selected SVR models |
| `fresh_candidates.csv`, `fresh_states.csv`, `fresh_instances.json` | Fresh evaluation inputs |
| `*_state_results.csv`, `evaluation_summary.csv`, `evaluation_summary.json` | State-level outcomes and aggregate measurements |
| `model_comparison.png` | Actual-gain comparison on both evaluation sets |

Both champion labels in the output files refer to the same SVR. They are retained to show that the prediction-error and decision-gain criteria agreed.

The reports were edited for readability after evaluation. `summarize.py` reads `report_text.md` so rebuilding the figure also preserves the revised wording. The original assignment deliverables remain the required Decision Tree/Random Forest comparison.
