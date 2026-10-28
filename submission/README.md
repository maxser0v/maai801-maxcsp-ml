# MAAI801 project package

## Files

- `completed_student.ipynb`: completed and executed notebook with saved outputs and interpretations.
- `report.docx`: editable eight-page report.
- `presentation.pptx`: nine editable slides with speaker notes and source citations.
- `contribution_statement.md`: AI assistance record and team table requiring actual names and contributions before submission.
- `candidate_dataset.csv`, `states.csv`, `instances.json`, `maxcsp_tools.py`: unchanged supplied inputs/helpers needed to run the notebook.
- `results/`: split manifest, tuning tables, frozen selection record, prediction metrics, state/instance comparisons, and summary JSON.
- `figures/`: two training EDA figures and two final comparison figures.
- `requirements.txt`: analysis packages used in the verified run.

## Run locally

1. Extract the package into one directory.
2. Open `completed_student.ipynb` in Jupyter using Python 3.
3. Install packages from requirements.txt if needed, then restart the kernel and run all cells in order.
4. Outputs are saved to `results/` and `figures/` in the current directory. Fixed seeds and instance grouping make the declared experiment reproducible.

## Run in Google Colab

Upload `completed_student.ipynb`, then upload the original instructor student ZIP in the Colab Files panel, or upload the four input/helper files directly. The first cell locates or extracts the data. Colab normally includes the analysis packages. Save the notebook with outputs after execution. Exact results can vary slightly with package versions; the recorded versions are in final_summary.json and requirements.txt.

## Result and scope

Random Forest was selected using validation MAE, 0.514774 versus 0.518270 for the Decision Tree. Both frozen configurations were then refit on all supplied training instances. Test MAEs were 0.436549 and 0.443934 respectively.

On 200 held-out states, mean actual improvement was 1.690 for ML and 1.565 for the heuristic. ML won 23 times, tied 172, and lost 5. This is a descriptive one-step comparison on synthetic instances, with no full genetic algorithm or runtime claim.

## Before submitting

Complete the contribution statement with real team details. Review all explanations and course AI-use requirements. Confirm Moodle deadline, accepted file formats, and whether supporting data are required. All four students should rehearse and understand the experiment. Nothing has been uploaded to Moodle.
