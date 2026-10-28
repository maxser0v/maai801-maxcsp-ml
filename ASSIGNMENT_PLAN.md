# MAAI801 assignment execution and delegation plan

## Primary goal
Deliver a reproducible notebook, a 6–10-page report, an 8–10-slide presentation, and an accurate contribution statement that answer whether supervised regression improves one-step variable selection over the supplied heuristic.

## Scope and source of truth
Read `MAAI801_Final_MiniProject.docx` for requirements and assessment. The supplied archive `MAAI801_MaxCSP_Student_Package.zip` contains the starter notebook, README, data, and evaluation helpers. Preserve these originals. No full Genetic Algorithm, reinforcement learning, new dataset, or guaranteed ML improvement is required.

All paths below are relative to this project directory. Notebook cell numbers are zero-based in the original notebook. The plan specifies work to delegate later; implementation has not started.

## Delegation and file ownership

| Owner | Main work | Owned outputs | Starts after |
|---|---|---|---|
| Coordinator | Setup, shared contract, integration, final evaluation, submission QA | completed_student.ipynb, results/final_*, submission/ | Phase 1; integrates after Phase 3 |
| Agent A | Data exploration and grouped validation split | work/data_exploration.py, work/eda_notes.md, results/split_manifest.csv, figures/eda_* | Phase 0 |
| Agent B | Decision Tree training and tuning | work/decision_tree.py, work/decision_tree_notes.md, results/dt_validation.csv | Phase 1 |
| Agent C | Random Forest training and tuning | work/random_forest.py, work/random_forest_notes.md, results/rf_validation.csv | Phase 1; parallel with B |
| Agent D | Fair comparison implementation, report and slides draft | work/compare_selection.py, work/report_draft.md, work/slides_outline.md | Phase 1; final results after Phase 4 |

Use one coordinator and up to three specialist agents at once if only four concurrency slots are available. Agent A can finish before Agent D starts. Only the coordinator edits the final notebook. Specialists deliver functions, tables, figures, and explanatory text for integration. Agents must not overwrite one another’s files or modify supplied evaluation helpers.

## Phase 0 — Read and verify the provided materials

**How this serves the primary goal:** Establishes the exact requirements and supplied evaluation behavior before anyone builds on assumptions.

What to do:
- Extract the archive into its supplied directory; preserve the ZIP and original assignment.
- Read the assignment, README, all starter notebook markdown/TODOs, and relevant helper implementations.
- Record actual function signatures and result dictionary fields from `maxcsp_tools.py` before using them.
- Establish Python 3 with pandas, NumPy, scikit-learn, matplotlib, and Jupyter or Google Colab.

References: assignment Parts 2–4; archive README sections “Included materials,” “Key definitions,” and “Technical requirements”; starter cells 0–4.

Verification:
- Setup cell locates all required files; baseline demonstration runs on a training state.
- Confirm 8,000 candidates, 1,000 states, 200 instances; 6,400 train candidates/160 instances and 1,600 test candidates/40 instances.
- Confirm all six feature names and target semantics against the helper code.

Guards: do not regenerate data unnecessarily, replace helpers, invent helper fields, or assume these are the original paper’s experiments.

### Verified helper API reference (inside supplied archive)
Discovery inspected all notebook cells and all 121 lines of `maxcsp_tools.py`.

| API | Source line |
|---|---|
| `load_instances(path)` | 119 |
| `n_violations(instance, assignment)` | 40 |
| `satisfaction(instance, assignment)` | 44 |
| `candidate_features(instance, assignment, var_index)` | 48 |
| `best_value_mutation(instance, assignment, var_index)` | 65 |
| `heuristic_select(instance, assignment)` | 92 |
| `evaluate_heuristic(instance, assignment)` | 98 |
| `evaluate_ml_selection(instance, assignment, model, feature_columns)` | 103 |

Mutation returns `selected_variable`, `old_value`, `new_value`, `violations_before`, `violations_after`, `satisfied_before`, `satisfied_after`, `improvement`, and the resulting assignment; ML evaluation also returns `predicted_improvement`. Verify exact assignment-result key when consuming it; comparison only needs the scalar metrics. The mutation helper copies its input. Heuristic ties use percentage, violation count, then lowest index; ML prediction ties use lowest index. Preserve these supplied rules. Keep notebook variables `fit_df`, `val_df`, `dt_model`, `rf_model`, and `best_model` for straightforward integration.

## Phase 1 — Establish the shared data and validation contract

**How this serves the primary goal:** Makes every model comparison reproducible and prevents leakage between related examples.

Owner: Agent A; coordinator approves the resulting contract before B/C start.

What to implement:
- Copy setup/loading and `FEATURES`/`TARGET` definitions from starter cells 1–2.
- Copy the `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` pattern from starter cell 8, using only supplied training instances and grouping by `instance_id`.
- Export a split manifest mapping each instance to fit, validation, or supplied test; reuse it for both models.
- Explore only training data: missing values, duplicate keys, ranges, target distribution, and state/candidate relationships. Do not silently discard legitimate repeated feature vectors.
- Produce at least two interpreted EDA figures, such as target distribution and observed improvement versus template percentage.
- Define a common model interface: `train_and_validate(fit_df, val_df, features, target)` returns the selected fitted model, a validation-results table, and its selected parameters. B/C may implement this in separate modules.

Contract:
```python
FEATURES = ['variable_degree', 'variable_violations', 'template_percentage',
            'current_violations', 'current_satisfaction', 'num_constraints']
TARGET = 'improvement'
```
Expected groups with this split: 128 fit instances and 32 validation instances; supplied test remains 40 instances. Expected candidate rows: 5,120 fit, 1,280 validation, 1,600 test. Each instance has five states and eight candidates per state.

References: starter cells 2, 5–8; README “Key definitions”; assignment Sections 2.2 and 3.6.

Verification: fit/validation/test instance IDs are pairwise disjoint; each state’s eight candidates stay together; targets are nonnegative; feature selection excludes IDs, split labels, target, and post-mutation information. Figure axes and explanatory notes are present.

Guards: no random row split; no test-data EDA used to guide modeling; no target-derived features; no independent split choices by B/C.

## Phase 2 — Train and tune the Decision Tree

**How this serves the primary goal:** Provides an interpretable supervised model and establishes its predictive performance without using test outcomes.

Owner: Agent B. Can run alongside Phase 3 after Phase 1.

What to implement:
- Follow `DecisionTreeRegressor` and `mean_absolute_error` imports and fit pattern in starter cell 8.
- Evaluate a small declared grid, for example `max_depth` in [3, 5, 8, None] and `min_samples_leaf` in [1, 5, 10], fixing `random_state=42`.
- Record all parameter combinations and validation MAE. Choose minimum MAE; resolve exact ties by simpler tree, then larger leaf size.
- Return the selected model fitted on fit data, parameters, validation table, and a short interpretation of under/overfitting.

References: starter cells 7–8 and assignment Section 2.3. Check installed scikit-learn documentation/signatures if additional APIs are needed.

Verification: predictions have one value per validation candidate; MAE is finite; all runs use the shared six features and shared partitions; selected parameters match the best eligible validation row.

Guards: no test prediction, test score, or hyperparameter selection based on test data. Avoid unnecessary scaling or model complexity.

## Phase 3 — Train and tune the Random Forest

**How this serves the primary goal:** Tests whether an ensemble improves predictive accuracy over the single tree using the same evidence.

Owner: Agent C. Can run alongside Phase 2.

What to implement:
- Follow `RandomForestRegressor` in starter cell 10 and the common interface from Phase 1.
- Evaluate a modest declared grid, for example `n_estimators` in [100, 200], `max_depth` in [5, None], and `min_samples_leaf` in [1, 5], fixing `random_state=42`.
- Record parameter settings and validation MAE; resolve exact ties by fewer trees, shallower depth, then larger leaves.
- Return selected fitted model, validation table, parameters, and short interpretation. Compare with the Decision Tree after both finish.

References: starter cells 9–10; assignment Section 2.3. Read installed scikit-learn documentation if using options beyond the starter.

Verification: same fit/validation instance IDs, features, target, and MAE calculation as Phase 2; finite results and reproducible selection.

Guards: no test access for tuning, no expansive search unrelated to grading, no selecting a model because it happens to beat the heuristic on test states.

## Phase 4 — Lock models and perform final evaluation

**How this serves the primary goal:** Produces an unbiased answer to both prediction accuracy and actual variable-selection quality.

Owner: coordinator, using comparison code from Agent D.

What to implement:
1. Combine validation tables and select the final model family using validation MAE alone. Declare tie rule in advance: prefer Decision Tree on an exact tie.
2. Freeze parameters and selection; save a short decision record before evaluating test data.
3. Refit each family’s selected configuration on all 160 supplied training instances. Retain original validation MAEs; evaluate both refitted models once on supplied test candidates to fill the assignment’s two-model table. `best_model` is the already selected family, even if the other family later has lower test MAE.
4. Follow starter cell 12: iterate only test states, parse `assignment_json`, retrieve the matching instance, and call `evaluate_heuristic` and `evaluate_ml_selection` with the same original state and same provided best-value operator. Use fresh assignment copies defensively.
5. Store state ID, instance ID, selected variables, actual gain, and violations after modification for both methods in `results/final_state_comparison.csv`.
6. Report mean actual improvement, mean remaining violations, number of states, and ML win/tie/loss counts and percentages. Also report how often chosen variables differ.
7. Produce prediction-error and actual-gain comparison figures. Complete starter cell 13 reflection questions using measured results.

References: starter cells 11–13; helper evaluation functions; assignment Sections 3.6 and 4.3.

Verification:
- Exactly 200 test states and 1,600 test candidates are evaluated; each state appears once.
- Wins + ties + losses = 200; percentages sum to 100% within rounding.
- For each method, gain equals original violations minus remaining violations and is nonnegative.
- Both methods start with identical violations; exported aggregate metrics reproduce directly from the saved state table.
- Results distinguish lower MAE from better decisions; no retuning after test results.

Guards: no sequentially feeding heuristic-modified states into ML; no repeated search trajectory; no unsupported speed claims. Timing is optional and only claimed if actually measured including feature construction and ML inference.

## Phase 5 — Assemble notebook, report, slides, and contributions

**How this serves the primary goal:** Turns verified experiments into clear, complete assessment deliverables.

Owners: coordinator integrates notebook; Agent D drafts narrative and slides; humans verify authorship and contribution claims.

What to implement:
- Build `submission/completed_student.ipynb` by retaining the supplied tasks and filling TODOs with specialist work, outputs, and interpretations. Keep paths portable for local Jupyter/Colab.
- Create a 6–10-page report: introduction/research distinction; data and grouped validation; models/tuning; prediction results; fair heuristic comparison; discussion/limitations; conclusion; references. Target roughly 1 page introduction, 1–2 data/methods, 1 models, 2 results, 1–2 discussion/references; verify actual page count after rendering.
- Include at least three meaningful labeled figures across notebook/report. With two EDA figures plus prediction and decision comparisons, there is a safe margin above the minimum.
- Prepare 8–10 slides: question; CSP example/paper distinction; data; leakage-safe split; two models/tuning; prediction results; selection results; limitations; conclusion/team. Target 12 minutes plus 5 minutes Q&A with all four students presenting.
- Draft a contribution table from actual work logs. Do not invent human work or claim agent-written content was independently produced by students. Follow course AI-use rules if provided.
- Cite the assigned 2015 paper accurately; read it before discussing details beyond the assignment summary. Explain that this project uses synthetic classroom instances and a simplified one-step procedure.

References: assignment Sections 4.1–4.5 and starter cell 13. Use relevant document/presentation skills when producing final formatted files.

Verification: all required tables present; same numbers in notebook/report/slides; three or more explained figures; report 6–10 rendered pages; slides 8–10; accurate contribution statement; all students can explain the overall method.

Guards: no invented results/references, unsupported claims about the full GA, or claim that negative comparative results mean project failure. Report an honest null/negative finding if that is what the evidence shows.

## Phase 6 — Final verification and submission package

**How this serves the primary goal:** Ensures the submitted work is runnable, consistent, and meets every assessed requirement.

Owner: coordinator; optionally delegate an independent read-only review after integration.

What to do:
- Restart the runtime and run the notebook from top to bottom using the provided files; confirm no hidden state, incomplete TODOs, errors, or absolute personal paths.
- This final reproducibility run repeats the frozen experiment only; it must not drive further test-based tuning.
- Recheck instance disjointness, feature whitelist, frozen model selection, fair same-state evaluation, table totals, and figure captions.
- Render and inspect report/slides; fix overflow, unreadable plots, inconsistent rounding, and missing references.
- Check every assignment checklist item and rubric component. Record unresolved limitations candidly.
- Package notebook, report, slides, contribution statement, and required supporting data/helpers or clear setup instructions. Confirm Moodle deadline, course link, and accepted formats with the course announcement; do not upload without a request.

References: assignment final checklist and rubric; original README setup instructions.

Acceptance: a fresh reader can run the notebook, reproduce the reported tables, identify how the model was selected, and understand whether ML improved one-step decisions.

## Copy-ready delegation prompts

Prepend this shared instruction to every specialist prompt:
> Work in the MAAI801 Final Project workspace. Read ASSIGNMENT_PLAN.md and the relevant source sections before coding. Preserve original data/helpers. Write only your owned files; do not edit the final notebook or another agent’s files. Use the shared split and six-feature contract. Do not inspect test outcomes or tune on test data. Report files created, commands/checks run, measured findings, source references, and remaining gaps. Never fabricate results.

**Agent A:** Execute Phase 1 after verifying Phase 0 setup. Deliver the split manifest, reusable exploration code, at least two saved EDA figures, and concise interpretation. Include assertions for grouping and expected sizes. Tell the coordinator when the shared contract is ready so model agents can start.

**Agent B:** Execute Phase 2 using Agent A’s manifest. Deliver a Decision Tree training function, all validation results and selected configuration, and concise model notes. Do not run final test evaluation.

**Agent C:** Execute Phase 3 using Agent A’s manifest and the same interface as Agent B. Deliver Random Forest training code, all validation results and selected configuration, and concise notes. Do not run final test evaluation.

**Agent D:** Read Phase 4 and the actual helper APIs, then implement a reusable same-state comparison function without running it on test data until the coordinator locks the model. Prepare report/slides structures with clearly marked result placeholders. After Phase 4 provides verified outputs, fill narrative and figures from those outputs and complete Phase 5 drafts.

**Coordinator:** Execute phases in dependency order: 0 → 1 → (2 and 3 in parallel; D prepares comparison) → 4 → 5 → 6. Own final model lock, test execution, integration, and verification. Ask specialists to resolve mismatches using validation evidence; do not select or retune from test performance.

## Required handoff evidence
Each agent returns source files/sections consulted; owned output paths; reproducible run instructions; verified results or assertions; assumptions and gaps. Use results files as the authoritative source for reported numbers. If a specialist hands off only prose or unsupported conclusions, request concrete artifacts and evidence before integrating.
