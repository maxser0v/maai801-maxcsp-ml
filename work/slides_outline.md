# Presentation outline — Max-CSP variable selection

**Target:** 9 slides, approximately 12 minutes plus 5 minutes for questions. All four students present. Results below are taken from the frozen evaluation artifacts.

| Slide | Time | Speaker | Content and visual |
|---|---:|---|---|
| 1. Research question | 0:45 | Student 1 | “Can regression improve one-step variable selection over a percentage heuristic?” State the scope in one sentence: synthetic classroom Max-CSPs and one mutation. Visual: a simple question/title graphic. |
| 2. CSP, Max-CSP, and paper context | 1:20 | Student 1 | Explain variables, domains, constraints, and maximizing satisfied constraints with the assignment's three-exam example. Briefly distinguish the 2015 paper's percentage template inside a GA from this project's trained ML extension. Visual: three variables and violated/satisfied edges. Cite Ben Othman & Tagina (2015), DOI 10.1016/j.procs.2015.08.156. |
| 3. Data and prediction target | 1:10 | Student 1 | 200 synthetic instances; 8 variables/domain size 4; 8,000 candidate rows; six pre-mutation features; target is reduction in violations after best-value mutation. Mention five states and eight candidates per state. Visual: one state → eight candidate rows → target. |
| 4. Leakage-safe evaluation | 1:15 | Student 2 | 160 train instances / 40 separate test instances. Group-wise split within train by `instance_id`, `test_size=0.2`, seed 42: 128 fit instances (5,120 rows), 32 validation instances (1,280 rows), and 40 test instances (1,600 rows). Explain related states remain together. Visual: fit/validation/test groups. |
| 5. Models and model choice | 1:20 | Student 2 | Decision Tree grid: depth {3, 5, 8, None} × leaf size {1, 5, 10}, 12 configurations. Random Forest grid: trees {100, 200} × depth {5, None} × leaf size {1, 5}, 8 configurations. Both use random_state 42 and shared features/split. Select final family by validation MAE; exact tie rule prefers Decision Tree. Visual: selected settings and validation MAE. |
| 6. Prediction performance | 1:10 | Student 2 | Decision Tree validation/test MAE: 0.518270 / 0.443934. Random Forest: 0.514774 / 0.436549. Random Forest selected on validation MAE. Explain MAE is candidate-level prediction error, not direct selection quality. Visual: labeled MAE comparison by split. |
| 7. Fair one-step comparison | 1:20 | Student 3 | On each same original state, heuristic chooses highest template percentage; ML chooses highest predicted gain; both use the same best-value mutation. Visual: two selection paths branching from one shared assignment and reconverging at the same evaluator. |
| 8. Actual selection results | 1:45 | Student 3 | On 200 same-state comparisons, mean gain was 1.565 for heuristic and 1.690 for ML (+0.125, about 8.0%). Mean violations after: 6.065 and 5.940. ML outcomes: 23 wins (11.5%), 172 ties (86.0%), 5 losses (2.5%); different variables in 40 states (20%). Across 40 instances: 18 better mean gains for ML, 20 ties, 2 lower. Visual: outcome counts and paired gain summary. |
| 9. Limits and conclusion | 1:15 | Student 4 | State conclusion narrowly: modest average gain in this synthetic one-step experiment, with most states tied. Limits: synthetic classroom data, one-variable best-value move, no complete GA/search trajectory, no runtime claim or transfer to paper instances. Close with the research answer and invite questions. |

## Suggested speaker transitions

- Student 1: slides 1–3 (context and data).
- Student 2: slides 4–6 (validation and models).
- Student 3: slides 7–8 (comparison and results).
- Student 4: slide 9 and facilitate questions. All members should be prepared to explain the shared method and results.

## Presentation consistency checklist

- Use only the coordinator's locked model and verified final artifacts.
- Keep validation MAE and test MAE clearly distinguished; model family is selected by validation MAE.
- Ensure the number of evaluated states equals wins + ties + losses; percentages total 100% after rounding.
- Use the same rounding and values as the report and notebook.
- Keep figure axes, state counts, and units visible.
- Do not claim a full-GA improvement, paper replication, or speedup from this one-step experiment.
