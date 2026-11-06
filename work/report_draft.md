# Machine learning guided variable selection for Max CSP

## Abstract

Choosing which variable to change can make a difference when trying to repair a solution that breaks several constraints. This project compares a percentage-based selection rule with a learned alternative. We trained Decision Tree and Random Forest regressors to predict how many violations a change would remove, using six measurements available before the change. We kept problem instances separate during validation and chose the Random Forest because it had the lower validation mean absolute error (MAE): 0.5148, compared with 0.5183 for the tree. Both chosen configurations were then refit on all 160 training instances. On 200 held-out states, the forest removed an average of 1.690 violations per change, while the heuristic removed 1.565. ML did better on 23 states, matched the heuristic on 172, and did worse on 5. The average advantage was small but positive on this test set. The experiment concerns a single change to a synthetic problem, so it does not establish how the method would perform inside a complete genetic algorithm.

## 1. Introduction

Consider an exam timetable in which two exams that share students have been placed at the same time. Moving an exam might resolve that clash, but it could also create a different one. The practical question is which exam to move first. Maximal Constraint Satisfaction Problems, or Max-CSPs, express this kind of problem using variables, possible values, and constraints. When all constraints cannot be satisfied, the aim is to satisfy as many as possible. Counting the constraints that remain violated gives a straightforward measure of solution quality.

Ben Othman and Tagina (2015) proposed using the percentage of a variable's constraints that are violated to guide a Genetic Algorithm. The motivation is easy to see. Two variables might each appear in two violated constraints, but one may belong to ten constraints in total and the other to only two. Their violation counts are equal, while their percentages are 20% and 100%. The percentage captures a distinction that a raw count misses.

A high percentage still does not tell us whether a useful replacement value exists. Changing the variable with the highest percentage may therefore produce less improvement than changing another variable. We wanted to find out whether a regression model could learn this difference from examples.

Our research question was: can supervised learning predict the benefit of changing a variable and use that prediction to improve selection over the percentage rule? We replaced only the choice of variable. After that choice, both methods used the same supplied procedure to find its best value. This let us compare selection methods without changing the rest of the operation.

The original paper used a predefined template within a Guided Genetic Algorithm. Our experiment adds supervised learning to a simplified one-step comparison. We evaluated both prediction error and actual improvement, because an accurate numerical prediction does not automatically lead to the best choice among competing variables.

## 2. Data and problem setup

### 2.1 What each example represents

The supplied data contain 200 synthetic binary Max-CSP instances. Each instance has eight variables, four possible values per variable, and five starting assignments, which we call states. A state contributes eight candidate rows, one for each variable that could be changed. Altogether, the dataset contains 8,000 rows. The supplied training partition has 6,400 rows from 160 instances; the separate test partition has 1,600 rows from 40 instances.

For each candidate, the model receives six inputs: the number of constraints involving the variable (`variable_degree`), its current violated constraints (`variable_violations`), their percentage (`template_percentage`), the whole assignment's violated and satisfied constraints (`current_violations` and `current_satisfaction`), and the total number of constraints (`num_constraints`). All six are measured before changing the variable. Instance IDs, state IDs, split labels, and the target are excluded from the inputs.

The target, `improvement`, is the number of violations before a change minus the number afterward. The supplied best-value procedure tries the variable's allowed values and keeps the best strict improvement. If no value helps, it retains the original value. A candidate can therefore have zero improvement, but never a negative target. The files `states.csv` and `instances.json` provide the assignments and constraints needed to check these outcomes.

### 2.2 Preparing and exploring the data

We divided the 160 training instances into 128 fit instances and 32 validation instances, giving 5,120 and 1,280 candidate rows respectively. We used `GroupShuffleSplit` with one split, a validation fraction of 0.2, and seed 42, grouping by `instance_id`. All five states and all candidate variables from one instance stayed in the same partition. A random split of individual rows would allow examples from the same underlying problem to appear on both sides, making the evaluation less informative. The exact membership is saved in `results/split_manifest.csv`.

Nearly half of the training candidates offered no improvement. The counts for gains of 0 through 5 were 3,077, 2,084, 911, 257, 67, and 4. Thus, 48.1% had a zero target, and large gains were uncommon. A useful model needs to distinguish the relatively promising moves from the many candidates that do little or nothing.

**Figure 1. Distribution of improvement in the training data.** Zero-gain candidates account for almost half of the examples, while gains above two are rare. Source image: `figures/eda_target_distribution.png`.

We found no missing training feature values, duplicate candidate keys, or exact duplicate complete rows. We retained repeated feature vectors: two different candidates can legitimately have the same measurements. Variable degree and incident violations ranged from 0 to 7, template percentage from 0 to 100, assignment violations from 0 to 25, satisfied constraints from 0 to 24, and total constraints from 3 to 28. Some inputs are related by construction; for example, the satisfied and violated counts add up to the total number of constraints.

The percentage rule has some support in the data, but its relationship with gain is uneven. Mean improvement was zero in the 0–10% band and about 1.61 in the 80–90% band, then fell to about 1.30 in the 90–100% band. These are averages over different candidates, rather than evidence that increasing a percentage causes a larger gain.

**Figure 2. Mean training improvement by template-percentage band.** Higher percentages often accompany larger gains, but the pattern is not monotonic. Source image: `figures/eda_improvement_by_template_percentage.png`.

## 3. Models and evaluation methods

### 3.1 Training the regressors

We used the two model families required by the assignment. A Decision Tree learns a series of splits on the inputs and predicts an improvement at each leaf. A Random Forest averages predictions from many trees. Both can represent nonlinear relationships without requiring us to specify those relationships in advance.

For the Decision Tree, we tried maximum depths of 3, 5, 8, and unrestricted depth, combined with minimum leaf sizes of 1, 5, and 10. These 12 configurations all used seed 42. The selected tree had a maximum depth of 8 and a minimum of 10 samples per leaf.

For the Random Forest, we tried 100 or 200 trees, maximum depth 5 or unrestricted depth, and minimum leaf size 1 or 5. These eight configurations used the same seed and exactly the same fit and validation groups as the tree. The selected forest had 200 trees, unrestricted depth, and a minimum leaf size of 5. Complete tuning results are saved in `results/dt_validation.csv` and `results/rf_validation.csv`.

We chose settings and then the final model family using validation MAE. MAE is the average absolute difference between predicted and actual improvement, expressed in violations per candidate. An exact tie between model families would have favored the Decision Tree. The forest's validation MAE was slightly lower, so we selected it before examining test performance. We then refit both selected configurations on all 160 training instances and measured their test errors.

### 3.2 Comparing the selected variables

The decision comparison used the original assignment for every test state. The heuristic chose the highest violation percentage, breaking ties by the number of incident violations and then by the lowest variable index. The forest predicted a gain for each of the eight variables and chose the largest prediction, breaking ties by the lowest index.

Both choices went through the same best-value mutation procedure. Each method received its own copy of the starting assignment, so neither benefited from the other's change. We recorded the selected variable, actual gain, and remaining violations.

An ML win means its selected move removed more violations than the heuristic's move. Equal gains count as a tie, even when the chosen variables differ. A loss means ML removed fewer violations. We also calculated mean gain, mean remaining violations, and the number of variable disagreements. We did not measure execution time, so the comparison makes no claim about speed.

## 4. Prediction results

### 4.1 Validation and test error

| Model | Selected settings | Validation MAE | Test MAE |
|---|---|---:|---:|
| Decision Tree | `max_depth=8`, `min_samples_leaf=10` | 0.518270 | 0.443934 |
| Random Forest | `n_estimators=200`, `max_depth=None`, `min_samples_leaf=5` | 0.514774 | 0.436549 |

The forest had the lower MAE on both partitions, although the validation difference was only about 0.0035 violations per candidate. That was enough to select it under our stated rule, but it is a small margin. A different validation sample could have changed the ordering.

The forest's test MAE of 0.436549 means that its predicted gains were about 0.44 violations away from the observed gains on average. The tree's test MAE was 0.443934. These scores describe numerical prediction accuracy; they do not tell us directly whether the highest-ranked variable was a good choice.

Both models had lower test error than validation error. We should interpret that difference cautiously. The test problems were a different sample, and the models used for testing had been refit on more training instances. The two scores therefore do not isolate the effect of the evaluation partition.

**Figure 3. Prediction error for the selected tree and forest configurations.** Test scores were measured after model selection and refitting. Source image: `figures/prediction_mae.png`.

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

The forest removed 0.125 more violations per state on average, an increase of about 8.0% relative to the heuristic's mean gain. Both methods started with an average of 7.630 violations. Subtracting their gains gives the remaining counts shown in the table: 6.065 for the heuristic and 5.940 for ML.

Most states produced the same gain under both methods. The forest did better in 23 states and worse in 5, while 172 were tied. It chose a different variable in 40 states, or 20% of the test set. Of those disagreements, 12 still led to equal gains. A different choice therefore did not always change the outcome.

The pattern was also positive when we averaged the five states within each problem instance. ML's mean gain was higher in 18 of the 40 instances, equal in 20, and lower in 2. The improvement was spread across several instances, but half still showed no average difference. The per-instance results are available in `results/final_instance_comparison.csv`.

**Figure 4. Actual gains and paired outcomes on the 200 test states.** The small increase in mean gain accompanies a large proportion of ties. Source image: `figures/selection_comparison.png`.

## 6. Discussion

The results support a modest benefit from learned selection on this test set. The percentage rule was already competitive: it matched the forest's actual gain on 86% of states. The forest's advantage came from a smaller set of decisions where it found a more useful variable, with relatively few decisions that were worse. This is a more precise description of the result than saying ML was simply better across the board.

Prediction error and decision quality need separate attention. MAE gives every candidate prediction equal weight, but a selector acts only on the candidate with the highest score. A model could predict most candidates well and still misorder the two best choices in a state. Conversely, imperfect estimates can be useful if they put a good candidate first. Our actual-gain comparison was necessary to show whether the predictions helped with the decision the assignment asks us to make.

The six inputs also place a limit on what the models can learn. They summarize counts and percentages, without describing every forbidden value pair or the values of neighboring variables. Two variables can share the same summaries and still have different opportunities for improvement. The forest can learn patterns in these summaries, but it cannot recover information that the inputs do not contain. We did not perform a feature-importance or ablation study, so we cannot attribute the observed advantage to a particular input.

The experiment has several other limits. The problems were small and synthetic, with eight variables and four values each. We used one grouped validation split and modest parameter grids. There were only 40 test instances, and the five states within an instance shared its constraint structure. The observed average is therefore useful evidence for this particular setting, rather than a reliable estimate for all Max-CSP applications.

Finally, a one-step gain does not tell us how a complete search would behave. A genetic algorithm maintains a population, selects parents, combines solutions, and makes repeated changes. We did not implement those stages or follow a sequence of states. Testing the selector inside such a process, on additional independently generated or real problems, would be a reasonable next step. It would require a new evaluation rather than an extrapolation from the present 8% figure.

## 7. Conclusion

The validation-selected Random Forest improved mean one-step gain from 1.565 to 1.690 violations removed on the supplied test states. It won more often than it lost, although most outcomes were ties. This answers the research question positively for the tested setup: the learned selector made a small average improvement over the percentage heuristic. The evidence remains limited to synthetic instances and a single shared mutation procedure; it does not establish a speed advantage or an improvement to a complete genetic algorithm.

## References

Ben Othman, H., & Tagina, M. (2015). Guided Genetic Algorithm: A New Template Concept. *Procedia Computer Science, 60*, 1101–1108. https://doi.org/10.1016/j.procs.2015.08.156.

MAAI801. (n.d.). *Machine Learning Guided Variable Selection for Maximal Constraint Satisfaction Problems: Mini-Project Proposal* [Course assignment handout].

## Appendix Reproducing the experiment

The notebook reads `candidate_dataset.csv`, `states.csv`, `instances.json`, and `maxcsp_tools.py` from the instructor's package. The split uses `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`, grouped by `instance_id`. It saves the split manifest, both tuning tables, the model-selection record, and the final state and instance comparisons under `results/`.

The verified run used Python 3.11.7, pandas 2.2.3, NumPy 2.2.1, scikit-learn 1.8.0, and matplotlib 3.10.0. The target is `improvement`, and the six predictor names are listed in Section 2. Running the completed notebook from a fresh kernel reproduces the declared workflow.
