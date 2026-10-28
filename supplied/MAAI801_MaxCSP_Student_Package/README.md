# MAAI801 — Student Resources | Max-CSP ML Mini-Project

## Start here

1. Read `MAAI801_MiniProject_Proposal.docx` (the project instructions and assessment rubric).
2. Open `student_starter.ipynb` in Jupyter or Google Colab.
3. **Google Colab:** Upload the notebook using **File → Upload notebook**. Then upload the *complete* ZIP `MAAI801_MaxCSP_Student_Package.zip` using the **Files** panel on the left. Run the first setup cell; it automatically extracts and finds the required files. There is no need to upload four CSV/Python files separately.
4. **Local Jupyter:** Extract the ZIP into a folder and launch the notebook from that folder. Run the notebook cells in order.
5. Complete the `TODO` sections. Keep the provided code and your finished notebook, and submit the deliverables in the proposal.

## Included materials

| File | What it is | Your responsibility |
|---|---|---|
| `MAAI801_MiniProject_Proposal.docx` | Short project instructions and grading criteria | Read carefully |
| `student_starter.ipynb` | Guided, incomplete programming tasks, setup and baseline demo | Complete and submit notebook |
| `candidate_dataset.csv` | 8,000 synthetic candidate-variable examples, each with 6 features and target `improvement` | EDA, training and evaluation |
| `states.csv` | 1,000 current assignments (5 per instance) | Test selection strategies on same states |
| `instances.json` | 200 synthetic Max-CSP definitions | Used by supplied tools |
| `maxcsp_tools.py` | Evaluator, percentage-based heuristic, best-value mutation, and ML comparison | Use; don't have to reimplement |
| `generate_dataset.py` | Optional reproducible data generation script | No need to run |
| `README.md` | This setup and dataset guide | Reference |

## Key definitions

- **Input columns:** `variable_degree`, `variable_violations`, `template_percentage`, `current_violations`, `current_satisfaction`, `num_constraints`.
- **Target column:** `improvement = violations_before - violations_after` (nonnegative here because the provided function retains the original value if no improving move exists).
- `template_percentage` is measured from **0 to 100**, not 0 to 1.
- Do not use IDs, `split`, or `improvement` as predictive features.
- Training set: **6,400 candidate rows from 160 problem instances**; test: **1,600 candidate rows from 40 different problem instances**.
- The five states belonging to any instance must stay in the same partition. Make a **group-wise validation split inside the train partition** using `instance_id`. Reserve the provided test partition for final evaluation only.
- Dataset is generated for teaching; **not the 2015 published research data**, and not an implementation of the full genetic algorithm.
- The original paper's template motivates a **simplified one-step baseline** that picks the variable with highest violation percentage (ties: more violations, then smaller index). Both the baseline and ML methods use the same `best_value_mutation` after choosing a variable.
- Students must evaluate **both** (1) regression prediction error, such as MAE, and (2) *actual reduction of violated constraints* when selected variables are modified.
- Do not claim runtime gains or improvements to the full Genetic Algorithm from this one-step classroom exercise.

## Technical requirements

Python 3 with pandas, numpy, scikit-learn, matplotlib. In Colab, these are usually already installed. If needed, run `%pip install pandas numpy scikit-learn matplotlib`.

**Reference research paper:** Ben Othman, H., & Tagina, M. (2015). *Guided Genetic Algorithm: A New Template Concept*. *Procedia Computer Science, 60*, 1101–1108. DOI: https://doi.org/10.1016/j.procs.2015.08.156
