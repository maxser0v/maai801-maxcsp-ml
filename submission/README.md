# MAAI801 project package

Start with `report.docx` for the main experiment, or open `completed_student.ipynb` to follow the code and results. The report compares the two required models, Decision Tree and Random Forest. The wider search across eight model families is described separately in `optional_extension/RESULTS.md`.

The reports have been revised for clearer wording. Their methods, measurements, and conclusions have not changed.

## Contents

| File or folder | Purpose |
|---|---|
| `completed_student.ipynb` | Executed notebook with the analysis, figures, and interpretations |
| `report.docx` | Editable main assignment report |
| `presentation.pptx` | Nine editable slides with speaker notes and citations |
| `contribution_statement.md` | Record of AI assistance and a table for the team's actual contributions |
| `candidate_dataset.csv`, `states.csv`, `instances.json`, `maxcsp_tools.py` | Original inputs and helpers needed to run the notebook |
| `results/` | Split manifest, tuning scores, selected settings, and final comparisons |
| `figures/` | Training exploration and final comparison plots |
| `requirements.txt` | Python analysis dependencies |
| `optional_extension/` | Additional-model report, comparison figure, and summary results |

## Run locally

1. Extract the package into one folder.
2. Install the packages in `requirements.txt` if needed.
3. Open `completed_student.ipynb` in Jupyter with Python 3.
4. Restart the kernel and run all cells in order.

The notebook writes outputs to `results/` and `figures/` in the current directory. The recorded random seeds and instance grouping make the experiment reproducible. The verified package versions are listed in `requirements.txt` and `results/final_summary.json`.

## Run in Google Colab

Upload the completed notebook, then upload the original instructor ZIP or the four input/helper files listed above. The setup cell locates or extracts the data. Colab usually includes the analysis libraries. Save a copy of the executed notebook, because uploaded runtime files may disappear after a reset. Small numerical differences are possible with different package versions.

## Reading the results

The main experiment selected the Random Forest using validation MAE: 0.514774, compared with 0.518270 for the tree. After refitting both configurations on all training instances, their test MAEs were 0.436549 and 0.443934 respectively.

On 200 test states, the forest removed an average of 1.690 violations per move, compared with 1.565 for the heuristic. It won 23 times, tied 172, and lost 5 times. The approximate 8% advantage describes this sample of synthetic states.

In the optional extension, SVR reached 7.03% above the heuristic on that original sample. On 2,500 fresh states, it reached 3.31%, compared with 1.76% for the original forest. The extension therefore found a useful alternative without exceeding the original 8% score on its original benchmark. Full reproduction scripts and data are available in the [GitHub repository](https://github.com/maxser0v/maai801-maxcsp-ml/tree/main/experiments/model_search).

## Before submission

Fill in the contribution statement with real team names and actual work. Review the explanations, check the course's AI-use requirements, and confirm the Moodle deadline and accepted formats. Each member should be able to explain the method and results. Nothing has been uploaded to Moodle.
