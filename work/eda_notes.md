# Phase 0/1: data and validation notes

## Supplied source and interpretation

The supplied `candidate_dataset.csv` has 8,000 candidate-variable rows from 200 synthetic Max-CSP instances. A row is one variable considered for a particular complete assignment (`state_id`); it contains six features known before mutation and `improvement`, the nonnegative number of violated constraints removed by the provided best-value mutation. There are eight variable candidates per state and five states per instance. The dataset provides 6,400 training rows from 160 instances and 1,600 held-out rows from 40 other instances.

Rows from one instance are related: the same constraint graph and variables recur across five states, and each state's eight candidate rows share the assignment-level violation/satisfaction values. A random row split could therefore put related examples in both fit and validation, making validation too optimistic. The shared split follows starter notebook cell 8 exactly: `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`, with the provided training rows only and `instance_id` as groups. This yields 128 fit instances (5,120 candidates) and 32 validation instances (1,280 candidates). The supplied test partition remains 40 instances (1,600 candidates). Each instance is assigned as a whole; split identities are written to `results/split_manifest.csv`.

## Training-only EDA

- Training target counts: 0: 3,077; 1: 2,084; 2: 911; 3: 257; 4: 67; 5: 4. Thus 48.1% of training candidates have zero improvement, with a long decreasing tail through five.
- Training feature ranges: variable degree 0–7; variable violations 0–7; template percentage 0–100; current violations 0–25; current satisfaction 0–24; constraints 3–28. The two assignment-level counts sum to the number of constraints for the source data, and are closely related by construction.
- There are no missing feature values, duplicate `(instance_id, state_id, variable_index)` keys, or exact duplicate full candidate rows in the training partition. Candidate rows were kept intact; legitimate repeated feature vectors were not removed or treated as errors.
- Training mean improvement generally rises across template-percentage bands, from 0 at 0–10% to about 1.61 at 80–90%, but it is not monotonic (for example, the 90–100% band averages about 1.30). The broad association makes the supplied percentage heuristic a plausible baseline, while the scatter in outcomes and nonmonotonic band means show that percentage alone does not determine the realized benefit. Prediction quality and variable-selection quality still require separate evaluation.

The figures use training instances only. Figure 1 shows the right-skewed, zero-heavy target distribution. Figure 2 summarizes mean realized improvement by pre-mutation template-percentage band and includes the number of candidates per band in the generated EDA summary printed by the script.

## Feature and helper contract

Use `FEATURES = ['variable_degree', 'variable_violations', 'template_percentage', 'current_violations', 'current_satisfaction', 'num_constraints']` and `TARGET = 'improvement'`. Identifiers (`split`, `instance_id`, `state_id`, `variable_index`) and post-mutation results are not model features. Repeated feature vectors are valid observations and have not been dropped.

`work/data_exploration.py` exports `load_data(project_root)`. It returns a dictionary with `fit_df`, `val_df`, `test_df`, `train_df`, `data`, `states`, `instances`, `manifest`, `features`, `target`, and `project_root`. Model workers should use `bundle = load_data(root)`, then `bundle['fit_df']`, `bundle['val_df']`, the module-level `FEATURES`, and `TARGET`. The `test_df` is present for later coordinator use only. Run with:

```sh
python3 work/data_exploration.py
```

The supplied helper API in `maxcsp_tools.py` (all inspected; line numbers from the extracted source) is: `n_violations(instance, assignment)` (40), `satisfaction(instance, assignment)` (44), `candidate_features(instance, assignment, var_index)` (48), `best_value_mutation(instance, assignment, var_index)` (65), `heuristic_select(instance, assignment)` (92), `evaluate_heuristic(instance, assignment)` (98), `evaluate_ml_selection(instance, assignment, model, feature_columns)` (103), and `load_instances(path)` (119). `best_value_mutation` returns `selected_variable`, `old_value`, `new_value`, `violations_before`, `violations_after`, `improvement`, `satisfied_before`, `satisfied_after`, and `new_assignment`. The ML evaluator adds `predicted_improvement`. It copies the input assignment and applies the same deterministic best-value mutation after selection.

## Sources consulted

- `ASSIGNMENT_PLAN.md`, Phase 0/1.
- `MAAI801_Final_MiniProject.docx` and the extracted `supplied/MAAI801_MaxCSP_Student_Package/MAAI801_MiniProject_Proposal.docx`, especially Sections 2.2, 2.3, 3.1–3.2, and 3.6.
- `supplied/MAAI801_MaxCSP_Student_Package/README.md`, including dataset organization and key definitions.
- All markdown/code cells in `supplied/MAAI801_MaxCSP_Student_Package/student_starter.ipynb`, especially setup/data cells 1–2, baseline cells 3–4, and grouped split cell 8.
- `supplied/MAAI801_MaxCSP_Student_Package/maxcsp_tools.py` and the supplied CSV headers/structural grouping.
