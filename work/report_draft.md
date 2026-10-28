# Can Regression Improve One-Step Variable Selection for Max-CSP?

**MAAI801 Mini-Project — report draft**

> Results below come from the coordinator's frozen model-selection record and final comparison CSVs. No tuning or evaluation was rerun while preparing this report.

## Abstract

This project asks whether supervised regression can improve one-step variable selection in synthetic maximal constraint satisfaction problems (Max-CSPs). We trained Decision Tree and Random Forest regressors to predict the reduction in violated constraints produced by changing one variable with the supplied deterministic best-value mutation operator. A group-wise validation split kept all examples from each problem instance together, and validation mean absolute error (MAE) determined the final model family. The Random Forest was selected (validation MAE 0.5148 violations per candidate versus 0.5183 for the Decision Tree) and both locked configurations were refit on all 160 training instances. On 200 held-out starting states, the percentage-based heuristic and the selected ML model independently chose variables from the same original assignments and used the same mutation operator. ML's mean actual improvement was 1.690 violations per state, compared with 1.565 for the heuristic, a difference of 0.125; ML won on 23 states, tied on 172, and lost on 5. ML selected a different variable on 40 states. The results show a modest average gain for this one-step setting, with most states tied. They do not evaluate a complete genetic algorithm or the original paper's experimental instances.

## 1. Introduction

A constraint satisfaction problem (CSP) represents a problem with variables, domains of possible values, and constraints on combinations of values. In a Max-CSP, the objective is to satisfy as many constraints as possible when satisfying every constraint may not be possible. For a complete assignment, the number of violated constraints provides a simple measure of solution quality.

The project is motivated by the template concept described by Ben Othman and Tagina (2015). The assignment summarizes the paper's variable score as the proportion of constraints involving a variable that are currently violated. This can distinguish variables with the same raw number of violations but different numbers of incident constraints. The original research applied a predefined heuristic within a Guided Genetic Algorithm (GGA); it did not train the regression models studied here. The assignment notes that the paper reported solution-quality and CPU-time improvements on its own test instances. Those findings do not establish that a classroom ML selector improves a full GA ([paper abstract and record](https://www.sciencedirect.com/science/article/pii/S1877050915022838)).

We study a narrower extension: replace only the variable-selection rule in a simplified one-step comparison. The heuristic chooses the variable with the largest violation percentage. The ML selector predicts the gain associated with changing each variable and chooses the highest predicted gain. Both choices are followed by the same supplied best-value mutation operator.

**Research question.** Can supervised learning predict the benefit of changing a variable and improve one-step variable selection compared with a percentage-based heuristic?

The study separates two questions. First, how accurately do the regression models predict candidate improvement? Second, do their selected variables yield larger actual reductions in violated constraints than the heuristic's choices? Lower prediction error does not necessarily imply better decisions because selection depends on the ranking of candidates within each state.

## 2. Data and problem setup

### 2.1 Synthetic instances and records

The supplied teaching data contain 200 generated binary Max-CSP instances, each with 8 variables and domain size 4. The candidate dataset has 8,000 rows: 6,400 from 160 training instances and 1,600 from 40 separate test instances. Each instance contributes five starting states; each state contributes one candidate row per variable, for eight candidate variables per state. The `states.csv` file provides the starting assignments and `instances.json` provides the constraints and forbidden value pairs. These are synthetic classroom instances, not the original paper's experimental data.

Each candidate row represents one variable considered for a possible one-variable change in a complete assignment. The six pre-mutation predictors are `variable_degree`, `variable_violations`, `template_percentage`, `current_violations`, `current_satisfaction`, and `num_constraints`. The target is `improvement`, defined as violations before the best-value mutation minus violations afterward. The supplied operator retains the current value if no strictly better value exists, so target values are nonnegative. Identifiers, split labels, and post-mutation information are excluded from the predictors.

### 2.2 Grouped validation design

The original training partition was split into fit and validation sets with `GroupShuffleSplit`, `n_splits=1`, `test_size=0.2`, and `random_state=42`, grouping on `instance_id`. This keeps all states and candidate variables from an instance on one side of the split, reducing leakage from examples that share an underlying constraint problem. The resulting partition contained 128 fit instances (5,120 candidate rows) for fitting and 32 validation instances (1,280 rows) for model selection. The separate 40-instance test partition (1,600 rows) was reserved for final evaluation. The exact assignment of instance IDs is in `results/split_manifest.csv`.

Training-only target counts were: 0 improvement, 3,077 candidates; 1, 2,084; 2, 911; 3, 257; 4, 67; and 5, 4. Thus, 48.1% of training candidates had zero improvement, and the distribution had a long, sparse tail. This imbalance makes a constant prediction near the common outcomes a meaningful reference point, while making larger gains relatively rare.

**Figure 1. Training-only candidate improvement distribution.** The outcome is concentrated at zero and decreases through five. Source image: `figures/eda_target_distribution.png`.

Training feature ranges were: variable degree 0–7; variable violations 0–7; template percentage 0–100; current violations 0–25; current satisfaction 0–24; and total constraints 3–28. No training feature values were missing, and there were no duplicate `(instance_id, state_id, variable_index)` keys or exact duplicate complete candidate rows. Repeated feature vectors were retained because distinct candidates can legitimately share the same pre-mutation measurements. The assignment-level counts for current violations and current satisfaction are linked by the total number of constraints, so these two predictors are not independent measurements.

**Figure 2. Training-only mean improvement by template-percentage band.** Mean improvement generally rises from 0 in the 0–10% band to about 1.61 in the 80–90% band, but is not monotonic: the 90–100% band averages about 1.30. This broad association supports using percentage as a baseline, while the variation shows that percentage alone does not determine realized gain. Source image: `figures/eda_improvement_by_template_percentage.png`.

## 3. Models and evaluation methods

### 3.1 Regression models

We trained a Decision Tree Regressor and a Random Forest Regressor on the shared fit partition and six-feature input. Hyperparameters were selected using validation MAE. The Decision Tree grid was `max_depth` ∈ {3, 5, 8, None} and `min_samples_leaf` ∈ {1, 5, 10}, with `random_state=42` (12 configurations). The selected tree used depth 8 and leaf size 10. The Random Forest grid was `n_estimators` ∈ {100, 200}, `max_depth` ∈ {5, None}, and `min_samples_leaf` ∈ {1, 5}, also with `random_state=42` (8 configurations). The selected forest used 200 trees, unbounded depth, and leaf size 5. The full validation tables are in `results/dt_validation.csv` and `results/rf_validation.csv`.

The final family was selected using validation MAE alone; an exact tie was resolved in favor of the Decision Tree. The Random Forest's validation MAE was lower (0.514774 versus 0.518270), so it was selected. Both selected configurations were then refit on all 160 supplied training instances for final evaluation. This refit policy uses all available training instances after model selection while preserving validation scores as the selection evidence.

MAE is the mean absolute difference between predicted and observed candidate improvement. It measures prediction error in units of violated constraints, averaged across candidate rows. It does not directly measure whether the model ranks the eight candidate variables in a state well enough to choose the best one.

### 3.2 One-step selection comparison

For each evaluated state, we parsed its original assignment and evaluated both selectors independently from that same assignment. The heuristic applied the supplied percentage rule, with ties resolved by incident violation count and then lower variable index. The ML model predicted improvement for each of the eight variables and chose the largest prediction, with ties resolved by lower index. Each selected variable was passed to the same deterministic best-value mutation operator. We recorded starting violations, selected variable, actual improvement, and remaining violations for both methods.

We classify a state as an ML **win** when ML's actual improvement exceeds the heuristic's, a **tie** when they are equal, and a **loss** when it is lower. We also report how often the selectors chose different variables. Variable disagreement is descriptive: different choices can still yield the same actual gain. Mean improvement and mean remaining violations summarize decision outcomes across states. No runtime claim is made unless timing is measured and includes ML feature construction and inference.

## 4. Prediction results

### 4.1 Validation and final prediction error

| Model | Selected settings | Validation MAE | Test MAE |
|---|---|---:|---:|
| Decision Tree | `max_depth=8`, `min_samples_leaf=10` | 0.518270 | 0.443934 |
| Random Forest | `n_estimators=200`, `max_depth=None`, `min_samples_leaf=5` | 0.514774 | 0.436549 |

The final model was the Random Forest, selected because it achieved the lower validation MAE. Its test MAE was 0.436549. The Decision Tree's test MAE was 0.443934; this test comparison is descriptive and did not alter the validation-based choice. Both test MAEs were lower than their corresponding validation MAEs. The test models also used more training data after refitting, so the scores are not a comparison of identical fitted models across splits. Moreover, the test partition is a finite sample of 40 held-out generated instances; the difference alone does not prove that the models generalize better to a broader population than the validation split.

**Figure 3. Candidate-level prediction MAE by model and split.** Lower values indicate smaller absolute prediction error; test values were computed only after model selection was locked. Source image: `figures/prediction_mae.png`.

## 5. One-step decision results

| Metric | Percentage heuristic | ML-guided selection |
|---|---:|---:|
| Mean actual improvement | 1.565 | 1.690 |
| Mean violations after mutation | 6.065 | 5.940 |
| Number of states | 200 | 200 |

| ML outcome relative to heuristic | States | Percentage |
|---|---:|---:|
| Win | 23 | 11.5% |
| Tie | 172 | 86.0% |
| Loss | 5 | 2.5% |
| **Total** | **200** | **100%** |

Across the 200 states, mean improvement was 0.125 violations per state higher for ML (about an 8.0% increase relative to the heuristic mean). The corresponding mean remaining violations were 5.940 for ML and 6.065 for the heuristic, also a 0.125 difference in ML's favor. Both methods began each comparison from identical assignments, whose mean starting violation count was 7.630. The accounting therefore matches: 7.630 − 1.565 = 6.065 for the heuristic and 7.630 − 1.690 = 5.940 for ML.

The state-level outcome distribution is dominated by ties: 172/200 (86.0%). ML achieved a larger gain on 23 states (11.5%) and a smaller gain on 5 (2.5%). It selected a different variable in 40/200 states (20.0%); variable disagreement does not necessarily imply a different resulting gain, because alternatives can yield equal improvement under the best-value mutation. Aggregating the same five states within each held-out instance, ML's mean gain was higher on 18 of 40 instances (45%), equal on 20 (50%), and lower on 2 (5%). This instance-level view is consistent with the modest positive average, while showing that the average improvement is not uniform across all instances.

**Figure 4. Actual one-step improvement by selection strategy.** The plot compares realized reduction in violated constraints on the same 200 starting states. The high tie count and small overall mean difference should remain visible in interpretation. Source image: `figures/selection_comparison.png`.

## 6. Discussion

The prediction and decision results answer different questions. MAE averages the absolute error of all candidate-level predictions; selection quality depends on how well the model ranks the eight candidates within each state. A model can have lower MAE while choosing a suboptimal variable in a particular state; conversely, imperfect numerical predictions can still rank candidates usefully. Here, the selected Random Forest had lower validation and test MAE than the Decision Tree and a higher mean actual gain than the heuristic, but the gain improvement was small relative to the large number of tied states. The evidence therefore supports a modest average benefit in this evaluation, not a broad claim that ML is consistently better on every state.

In this experiment, ML-guided selection improved average one-step actual gain compared with the percentage heuristic by 0.125 violations per state: 1.690 versus 1.565. The state outcomes were 23 wins, 172 ties, and 5 losses. These results suggest that the six available pre-mutation features contain some useful signal beyond the heuristic's percentage ordering, but the analysis does not isolate which feature or combination produced the observed difference. The `current_violations` and `current_satisfaction` features are linked by construction, and the feature set does not include detailed identities of neighboring variables, individual violated constraints, or explicit candidate post-mutation outcomes. The learner predicts from aggregate summaries rather than a complete structural representation of the problem.

The exercise is a simplified one-step procedure. It does not implement a population, parent selection, crossover, or repeated search trajectory for a complete genetic algorithm. It therefore cannot support conclusions about full-GA solution quality or speed. Likewise, the supplied synthetic instances are not the instances used in the cited paper. The study's evidence is useful for comparing two local variable-selection rules under controlled, shared conditions, but its scope is intentionally limited.

Other limitations include the finite synthetic sample (160 training, 40 test instances), a single fixed grouped validation split, two tested model families, and a target defined by the supplied deterministic best-value operator. The training and test instances come from the supplied generator and should not be treated as independent evidence about every real-world Max-CSP distribution. The single validation split makes model selection sensitive to that partition, while the fixed feature set and limited grids constrain what patterns the models can learn. Broader claims would require new independently generated or real problem instances, repeated group-wise validation, and evaluation under the intended search process. Those extensions are outside this assignment.

## 7. Conclusion

We compared a Decision Tree and Random Forest for predicting one-variable improvement, then evaluated the validation-selected Random Forest against the percentage heuristic on identical starting states using the same best-value mutation. The Random Forest achieved test MAE 0.436549. On 200 held-out states, ML achieved mean actual improvement 1.690 and recorded 23 wins, 172 ties, and 5 losses against the heuristic's mean 1.565. Thus, for this one-step classroom setting, the evidence supports a modest improvement in average actual gain. It does not establish improvement for a complete genetic algorithm or reproduce the original paper's experiments.

## References

Ben Othman, H., & Tagina, M. (2015). Guided Genetic Algorithm: A New Template Concept. *Procedia Computer Science, 60*, 1101–1108. [https://doi.org/10.1016/j.procs.2015.08.156](https://doi.org/10.1016/j.procs.2015.08.156). [Publisher record](https://www.sciencedirect.com/science/article/pii/S1877050915022838).

MAAI801. (n.d.). *Machine Learning Guided Variable Selection for Maximal Constraint Satisfaction Problems: Mini-Project Proposal* [Course assignment handout].

## Appendix: reproducibility record

- Supplied inputs: `candidate_dataset.csv`, `states.csv`, `instances.json`, and `maxcsp_tools.py` from the instructor's student package
- Group split: `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`, grouped by `instance_id`
- Feature columns: `variable_degree`, `variable_violations`, `template_percentage`, `current_violations`, `current_satisfaction`, `num_constraints`
- Target: `improvement`
- Model selection rule: lowest validation MAE; exact tie prefers Decision Tree
- Final comparison outputs: `results/final_state_comparison.csv`, `results/final_instance_comparison.csv`, and `results/final_summary.json`
- Frozen evaluation environment: Python 3.11.7; pandas 2.2.3; NumPy 2.2.1; scikit-learn 1.8.0; matplotlib 3.10.0
