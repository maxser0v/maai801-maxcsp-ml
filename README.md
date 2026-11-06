# Machine learning guided variable selection for Max-CSP

This MAAI801 mini-project asks whether a learned model can choose a more useful variable to change than a simple percentage rule. Both methods start from the same assignment and use the same procedure to change the chosen variable. We compare how many constraint violations each move removes.

## Reports and deliverables

- [Main assignment report](submission/report.docx): the required Decision Tree and Random Forest comparison, with methods, figures, results, and discussion.
- [Additional model experiments](experiments/model_search/RESULTS.md): the wider search across eight model families and evaluation on fresh synthetic problems.
- [Completed notebook](submission/completed_student.ipynb): runnable code with saved outputs.
- [Presentation](submission/presentation.pptx): nine editable slides with speaker notes.
- [Beginner explanation](ASSIGNMENT_EXPLAINED.md): a plain-language introduction to the problem and workflow.
- [Downloadable package](MAAI801_Completed_Project.zip): the main deliverables, required inputs, and a copy of the optional experiment report and summary results.
- [Execution plan](ASSIGNMENT_PLAN.md): the original plan and division of tasks.

Both reports have been revised for clearer, more natural prose. The numerical results and experimental methods are unchanged.

## Main experiment

We tuned both required models on the same grouped validation split. All records from a problem instance stayed together. The forest had the lower validation error, so we selected it before testing and refit both chosen configurations on all training instances.

| Model | Validation MAE | Test MAE |
|---|---:|---:|
| Decision Tree | 0.518270 | 0.443934 |
| Random Forest | 0.514774 | 0.436549 |

On 200 states from 40 held-out instances, the forest removed 1.690 violations per state on average, compared with 1.565 for the percentage heuristic. It recorded 23 wins, 172 ties, and 5 losses. The mean advantage was 0.125 violations per state, about 8% above the heuristic's mean gain.

## Additional models

The optional extension tested 45 configurations across Decision Tree, Random Forest, Extra Trees, Histogram Gradient Boosting, Gradient Boosting, Support Vector Regression (SVR), K-nearest-neighbor regression, and Ridge regression. Four-fold cross-validation grouped by instance selected a standardized SVR.

| Improvement over the heuristic | Original forest | Selected SVR |
|---|---:|---:|
| Original 200 states, previously evaluated | 7.99% | 7.03% |
| Fresh 2,500 states from 500 instances | 1.76% | 3.31% |

SVR did not exceed the original 8% result on the original sample. It did perform better than the forest on the larger fresh sample. The [extension report](experiments/model_search/RESULTS.md) explains the selection process, uncertainty, and why percentages from different samples need careful interpretation.

Both experiments concern one-step moves on synthetic problems. They do not measure a complete Genetic Algorithm or establish a runtime advantage.

## Run the notebook

The verified run used Python 3.11. Create an environment and install the recorded analysis dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r submission/requirements.txt
jupyter notebook submission/completed_student.ipynb
```

Run the cells in order. The notebook finds the supplied inputs and saves figures and result tables in the current working directory. See the [package instructions](submission/README.md) for Google Colab setup.

The component scripts can also be run from the repository root:

```bash
python3 work/data_exploration.py
python3 work/decision_tree.py
python3 work/random_forest.py
python3 work/run_final_evaluation.py
```

These commands reproduce the declared main experiment. The final test results should not be used to choose new parameters. Instructions for the optional search are in its [README](experiments/model_search/README.md).

## Files in this repository

| Directory | Contents |
|---|---|
| `submission/` | Final assignment files and the inputs needed by the notebook |
| `supplied/` | The extracted instructor package, with its original helpers |
| `work/` | Analysis and document-building scripts, notes, and verification records |
| `results/` | Main-experiment split, tuning, model-selection, and evaluation records |
| `figures/` | Main-experiment charts |
| `experiments/model_search/` | Optional search code, fitted models, fresh data, scores, and report |

The report builder also needs `python-docx`. The slide authoring script uses the Codex-bundled `@oai/artifact-tool` and presentation skill resources; the exported deck can be opened independently in PowerPoint-compatible software.

## Attribution and submission status

The course supplied the assignment, data generator, synthetic data, starter notebook, and constraint/mutation helpers. AI assistance generated the implementation, analysis, and initial drafts. The [contribution statement](submission/contribution_statement.md) still needs the team's actual names and contributions before academic submission. Nothing has been submitted to Moodle.

The original course materials retain their existing ownership and terms. This repository does not assign them a new license.

Research motivation: Ben Othman, H., & Tagina, M. (2015). *Guided Genetic Algorithm: A New Template Concept*. Procedia Computer Science, 60, 1101–1108. [DOI](https://doi.org/10.1016/j.procs.2015.08.156).
