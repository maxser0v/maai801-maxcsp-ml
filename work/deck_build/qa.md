# Presentation build and QA

- Deliverable: `submission/presentation.pptx` (9 editable slides, 16:9).
- Runtime: bundled Node.js with `@oai/artifact-tool`; exported through the presentation finalizer.
- Finalizer: passed PPTX package integrity, slide geometry, native table ownership (slides 4, 5, 8), native chart ownership (slides 3, 6, 8), and import validation. The validated revision was copied byte-for-byte to the requested deliverable path.
- Visual check: rendered the final PPTX through Artifact Tool and inspected all nine slides at full resolution. Corrected label collision on slide 2 and a truncated horizontal-chart axis on slide 8 before the final pass. No remaining clipping or overlaps observed.
- Editability: four charts across slides 3, 6, 8 are native PowerPoint charts; tables on slides 4, 5, 8 are native tables; slide diagrams and text are editable shapes.
- Speaker notes: present on all nine slides. They include source citations, presenter guidance, timing, and a four-presenter division of sections (approximately 11 minutes 20 seconds total plus questions).
- Data source for final metrics: `results/final_summary.json`, `results/final_prediction_metrics.csv`, `results/final_state_comparison.csv`, and `results/final_instance_comparison.csv`. No model/test evaluation was run while assembling the deck.

## Values included

- Validation MAE: Decision Tree 0.518270; Random Forest 0.514774. The Random Forest was selected on validation MAE and then refit on all 160 training instances.
- Test MAE: Decision Tree 0.443934; Random Forest 0.436549.
- Test-state comparison: 200 states, 40 instances; mean gain 1.565 heuristic vs 1.690 Random Forest; remaining violations 6.065 vs 5.940; 23 wins, 172 ties, 5 losses; different variables in 40 states.
- Conclusion is limited to the synthetic one-step setting and makes no claim about a full genetic algorithm or runtime.

## Visual design

Arial, warm off-white canvas, dark navy text, teal and coral data accents. The deck uses editable data visualizations rather than decorative imagery.

- Application testing: finalizer re-imported the deck and the final PPTX was rendered through Artifact Tool; it was not opened in desktop PowerPoint.
