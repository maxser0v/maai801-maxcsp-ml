# Machine learning guided variable selection for Max-CSP

MAAI801 Foundation of AI mini-project comparing a percentage-based variable-selection heuristic with supervised regression on synthetic constraint satisfaction problems.

The experiment predicts the benefit of changing one variable, then evaluates whether the selected variable reduces more constraint violations. It studies one-step selection rather than a complete Genetic Algorithm.

## Results

Both model families were tuned on the same instance-grouped validation split. The Random Forest was selected by validation MAE before final testing; both selected configurations were then refit on all training instances.

| Model | Validation MAE | Test MAE |
|---|---:|---:|
| Decision Tree | 0.518270 | 0.443934 |
| Random Forest | 0.514774 | 0.436549 |

On 200 test states from 40 held-out instances:

| Metric | Percentage heuristic | ML selection |
|---|---:|---:|
| Mean violations removed | 1.565 | 1.690 |
| Mean remaining violations | 6.065 | 5.940 |

ML achieved 23 wins, 172 ties, and 5 losses. The observed average gain was 0.125 violations per state, approximately 8% above the heuristic's mean. This result is limited to the supplied synthetic one-step setting and makes no runtime or complete-GA claim.

## Start here

- [Simple explanation for beginners](ASSIGNMENT_EXPLAINED.md)
- [Completed notebook with saved outputs](submission/completed_student.ipynb)
- [Eight-page report](submission/report.docx)
- [Nine-slide presentation with speaker notes](submission/presentation.pptx)
- [Execution and delegation plan](ASSIGNMENT_PLAN.md)
- [Downloadable project package](MAAI801_Completed_Project.zip)

## Run the experiment

Python 3.11 was used for the verified run.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r submission/requirements.txt
jupyter notebook submission/completed_student.ipynb
```

Run all notebook cells in order. The notebook locates the supplied inputs in this repository and writes results and figures to the current working directory. For Colab instructions, see [submission setup](submission/README.md).

To reproduce the component workflow from the repository root:

```bash
python3 work/data_exploration.py
python3 work/decision_tree.py
python3 work/random_forest.py
python3 work/run_final_evaluation.py
```

The final command reruns the frozen experiment for reproducibility. Do not use the held-out test results to retune the models.

## Repository layout

- `submission/`: deliverables and self-contained notebook inputs.
- `supplied/`: original instructor starter package, extracted without modifying its helpers.
- `work/`: analysis modules, authoring scripts, notes, and verification records.
- `results/`: split manifest, tuning tables, model-selection record, and final measurements.
- `figures/`: training exploration and final comparison plots.

The notebook and ML scripts are reproducible using the listed Python dependencies. Report generation additionally uses `python-docx`. The presentation authoring script uses the Codex-bundled `@oai/artifact-tool` and presentation skill resources; exported slides can be opened independently in PowerPoint-compatible software.

## Attribution and project status

The course supplied the assignment, synthetic data, generator, starter notebook, and constraint/mutation helpers. AI assistance generated the implementation, analysis, and draft deliverables. The [contribution statement](submission/contribution_statement.md) still requires the team's actual names and contributions before academic submission. No submission to Moodle has been performed.

Original supplied course materials retain their existing ownership and terms; this repository does not assign them a new license.

Research motivation: Ben Othman, H., & Tagina, M. (2015). *Guided Genetic Algorithm: A New Template Concept*. Procedia Computer Science, 60, 1101–1108. [DOI](https://doi.org/10.1016/j.procs.2015.08.156).
