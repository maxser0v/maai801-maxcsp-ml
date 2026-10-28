import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../..');
const SKILL_DIR = process.env.CODEX_PRESENTATION_SKILL_DIR;
if (!SKILL_DIR) throw new Error('Set CODEX_PRESENTATION_SKILL_DIR to the installed Codex presentation skill directory.');
const RUNTIME_PYTHON = process.env.CODEX_ARTIFACT_PYTHON || 'python3';
const BUILD_DIR = HERE;
const STAGING = path.join(BUILD_DIR, 'finalizer-r2');
const FINAL_PPTX = path.join(ROOT, 'submission/presentation-checked-r2.pptx');
const SUMMARY = JSON.parse(await fs.readFile(path.join(ROOT, 'results/final_summary.json'), 'utf8'));

const C = {
  paper: '#F7F5EF', ink: '#152536', muted: '#526273', teal: '#117C7B',
  tealPale: '#DCEDE9', coral: '#CC6842', coralPale: '#F4E2D8',
  gold: '#D6A747', line: '#B8C5C8', white: '#FFFFFF', blue: '#315F82',
  gray: '#E5E9E7', dark: '#0D2935', green: '#4E8067', red: '#B34F4F',
};
const FONT = 'Arial';
const W = 1280, H = 720;
const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const { applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);

function text(slide, value, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: 'textbox', name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: 'none', line: { style: 'solid', fill: 'none', width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontSize: opts.size ?? 25, typeface: FONT, color: opts.color ?? C.ink,
    bold: opts.bold ?? false, alignment: opts.align ?? 'left',
    verticalAlignment: opts.valign ?? 'top', wrap: 'square',
    autoFit: 'shrinkText', insets: opts.insets ?? 0,
  };
  return shape;
}
function line(slide, x1, y1, x2, y2, color = C.line, width = 2) {
  return slide.shapes.add({
    geometry: 'line', position: { left: Math.min(x1,x2), top: Math.min(y1,y2), width: Math.abs(x2-x1), height: Math.abs(y2-y1), verticalFlip: (x2-x1)*(y2-y1) < 0 },
    fill: 'none', line: { style: 'solid', fill: color, width },
  });
}
function box(slide, x, y, w, h, fill, stroke = fill, radius = 8) {
  return slide.shapes.add({ geometry: 'roundRect', position: { left:x, top:y, width:w, height:h },
    fill, line: { style:'solid', fill:stroke, width:1.5 }, borderRadius:radius });
}
function circle(slide, cx, cy, r, fill, stroke = fill, width = 2) {
  return slide.shapes.add({ geometry: 'ellipse', position:{left:cx-r,top:cy-r,width:2*r,height:2*r},
    fill, line:{style:'solid',fill:stroke,width} });
}
function newSlide(title, page) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  text(slide, title, 72, 40, 1130, 64, { size: 44, bold: true, color: C.dark, name: `slide-${page}-title` });
  line(slide, 72, 112, 1208, 112, C.line, 1.25);
  text(slide, String(page).padStart(2,'0'), 1174, 680, 36, 20, { size: 14, color: C.muted, align: 'right' });
  return slide;
}
function notes(slide, speaker, time, body, citations=[]) {
  const cited = citations.length ? `\n\nSources\n${citations.map(x=>`• ${x}`).join('\n')}` : '';
  slide.speakerNotes.text = `${speaker} · ${time}\n\n${body}${cited}`;
}
function addChart(slide, kind, config) {
  const chart = slide.charts.add(kind, config);
  applyPresentationChartFont(chart, { fontFamily: FONT });
  return chart;
}
function styleTable(table, rows, cols) {
  table.borders.assign({ style: 'solid', fill: C.line, width: 1 });
  for (let r=0;r<rows;r++) for(let c=0;c<cols;c++) {
    const cell=table.getCell(r,c);
    cell.fill = r===0 ? C.dark : (r%2===0 ? '#EDF0EC' : C.white);
    cell.text.style = { fontSize: r===0?19:20, typeface:FONT,
      color:r===0?C.white:C.ink, bold:r===0, verticalAlignment:'middle',
      autoFit:'shrinkText', insets:{top:8,right:10,bottom:8,left:10} };
  }
}
function makeTable(slide, values, x,y,w,h, widths) {
  const table=slide.tables.add({ rows:values.length, columns:values[0].length,
    left:x, top:y, width:w, height:h, values, ...(widths?{columnWidths:widths}:{}) });
  styleTable(table, values.length, values[0].length);
  return table;
}
function p2(x) { return `${(x*100).toFixed(1)}%`; }
function f3(x) { return Number(x).toFixed(3); }
function paramsLabel(model) {
  if (model==='Decision Tree') return `depth ${SUMMARY.dt_params.max_depth}; leaf ${SUMMARY.dt_params.min_samples_leaf}`;
  return `${SUMMARY.rf_params.n_estimators} trees; unlimited depth; leaf ${SUMMARY.rf_params.min_samples_leaf}`;
}

// 1. Research question
{
  const s=presentation.slides.add(); s.background.fill=C.dark;
  text(s, 'MAAI801 · MAX-CSP MINI-PROJECT', 82, 68, 700, 28, { size:18, bold:true, color:'#A8D8D0' });
  text(s, 'Can regression improve\none-step variable selection?', 82, 155, 1050, 190,
    { size:60, bold:true, color:C.white, name:'cover-title' });
  text(s, 'Percentage heuristic  vs  Random Forest guidance', 84, 390, 1000, 50,
    { size:29, color:'#D6E2E2' });
  line(s,84,484,260,484,C.gold,5);
  text(s, 'A controlled comparison on synthetic classroom Max-CSP instances', 84, 510, 940, 46,
    { size:24, color:'#D6E2E2' });
  text(s, 'One selected variable · one shared mutation operator', 84, 566, 940, 38,
    { size:21, color:'#A9BFC2' });
  notes(s,'Student 1','0:45','Open with the research question. This project replaces only the percentage heuristic’s variable-selection rule with a regression model. The experiment uses synthetic classroom instances and one best-value mutation; it does not implement a complete genetic algorithm.',[
    'MAAI801 Mini-Project Proposal, Parts 1–3 (supplied course handout).',
    'MAAI801_MaxCSP_Student_Package/README.md, scope and key definitions.'
  ]);
}
// 2. CSP and research context
{
  const s=newSlide('Max-CSP and the 2015 paper context',2);
  text(s,'A complete assignment chooses one value for every variable. Max-CSP seeks to maximize satisfied constraints.',
    82,137,1080,58,{size:25,color:C.muted});
  // Diagram: variables and constraints, as an editable explanatory figure.
  const pts=[[180,335],[395,260],[395,420]];
  line(s,pts[0][0],pts[0][1],pts[1][0],pts[1][1],C.coral,5);
  line(s,pts[0][0],pts[0][1],pts[2][0],pts[2][1],C.teal,5);
  line(s,pts[1][0],pts[1][1],pts[2][0],pts[2][1],C.teal,5);
  pts.forEach((p,i)=>{circle(s,p[0],p[1],43,C.white,C.dark,2);text(s,`X${i+1}`,p[0]-30,p[1]-16,60,32,{size:24,bold:true,align:'center'});});
  text(s,'one violated constraint',91,474,228,30,{size:18,bold:true,color:C.coral,align:'center'});
  text(s,'two satisfied constraints',344,474,238,30,{size:18,bold:true,color:C.teal,align:'center'});
  line(s,650,166,650,572,C.line,1.5);
  text(s,'The paper',700,174,425,35,{size:27,bold:true,color:C.teal});
  text(s,'Ben Othman & Tagina (2015) use a percentage template inside a Guided Genetic Algorithm.',700,220,475,106,{size:24});
  text(s,'This project',700,354,425,35,{size:27,bold:true,color:C.coral});
  text(s,'We train regressors to predict one-step gain, then compare selected variables under the same mutation rule.',700,400,475,122,{size:24});
  text(s,'A trained selector is an extension for this assignment, not the paper’s method.',700,538,470,48,{size:20,bold:true,color:C.muted});
  notes(s,'Student 1','1:20','Use the assignment’s three-exam example: X1=AI, X2=Databases and X3=Machine Learning; domains are time slots 1–3. In assignment [1,1,2], one constraint is violated and two are satisfied. The 2015 paper used a predefined percentage template within a GA. Here, supervised regression predicts one-step improvement; we are not reproducing the paper’s GA experiments.',[
    'Ben Othman, H., & Tagina, M. (2015). Guided Genetic Algorithm: A New Template Concept. Procedia Computer Science, 60, 1101–1108. https://doi.org/10.1016/j.procs.2015.08.156',
    'MAAI801 Mini-Project Proposal, Sections 1.1–2.1.'
  ]);
}
// 3. Data and target
{
  const s=newSlide('Data and prediction target',3);
  text(s,'Each row represents one candidate variable in one complete starting assignment.',82,134,1100,38,{size:25,color:C.muted});
  addChart(s,'bar',{position:{left:72,top:190,width:535,height:394},title:'Improvement on training candidates',
    titleTextStyle:{fontSize:20,fill:C.dark,bold:true},categories:['0','1','2','3','4','5'],
    series:[{name:'Rows',values:[3077,2084,911,257,67,4],fill:C.teal}],
    hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:55},
    xAxis:{textStyle:{fontSize:16,fill:C.muted},line:{style:'solid',fill:C.line,width:1}},
    yAxis:{title:{text:'Candidate rows',textStyle:{fontSize:16,fill:C.muted}},textStyle:{fontSize:15,fill:C.muted},majorGridlines:{style:'solid',fill:'#DDE3E1',width:1}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:14,fill:C.ink}},chartFill:'none',plotAreaFill:'none'});
  addChart(s,'bar',{position:{left:642,top:190,width:560,height:394},title:'Mean observed improvement by template band',
    titleTextStyle:{fontSize:20,fill:C.dark,bold:true},categories:['0–10','10–20','20–30','30–40','40–50','50–60','60–70','70–80','80–90','90–100'],
    series:[{name:'Mean gain',values:[0,.42963,.665574,.726688,.774005,1.353571,.967462,1.202505,1.605769,1.298507],fill:C.coral}],
    hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:45},
    xAxis:{textStyle:{fontSize:13,fill:C.muted},line:{style:'solid',fill:C.line,width:1}},
    yAxis:{title:{text:'Violations reduced',textStyle:{fontSize:15,fill:C.muted}},textStyle:{fontSize:14,fill:C.muted},majorGridlines:{style:'solid',fill:'#DDE3E1',width:1}},
    chartFill:'none',plotAreaFill:'none'});
  text(s,'6 pre-mutation features → predict improvement after the supplied best-value move',82,614,1110,35,{size:22,bold:true,color:C.dark});
  text(s,'48.1% of training candidates yield zero gain; higher template bands trend upward, but not monotonically.',82,649,1080,27,{size:17,color:C.muted});
  notes(s,'Student 1','1:10','Explain that each of 200 synthetic instances has eight variables and domain size four. Each instance has five states and each state contributes eight candidate rows. Predictors are degree, incident violations, template percentage, current violations, current satisfaction, and number of constraints. The target is violations before minus violations after the same best-value mutation. The EDA shows a zero-heavy target, and mean gain by template band is not monotonic.',[
    'MAAI801_MaxCSP_Student_Package/README.md, Included materials and Key definitions.',
    'MAAI801 Mini-Project Proposal, Sections 2.2 and 3.1–3.2.',
    'Training-only summaries: work/eda_notes.md.'
  ]);
}
// 4. Group-wise split
{
  const s=newSlide('Leakage-safe validation split',4);
  text(s,'All rows from an instance stay together. The held-out test instances remain separate.',82,139,1080,45,{size:25,color:C.muted});
  makeTable(s,[['Partition','Instances','Candidate rows'],['Fit','128','5,120'],['Validation','32','1,280'],['Supplied test','40','1,600']],
    82,222,600,302,[250,160,190]);
  text(s,'Provided train set\n160 instances · 6,400 rows',756,222,400,76,{size:26,bold:true,color:C.dark});
  line(s,774,316,1140,316,C.teal,4);
  text(s,'GroupShuffleSplit\n80/20 · seed 42\nby instance_id',790,340,300,105,{size:24,bold:true,color:C.teal,align:'center'});
  text(s,'Validation selects model settings and family. The supplied test set is reserved for the final report.',756,485,430,86,{size:23,color:C.muted});
  notes(s,'Student 2','1:15','The supplied training partition contains 160 instances. GroupShuffleSplit with a 20% validation share and random_state 42 creates 128 fit groups and 32 validation groups. Grouping by instance_id prevents related states and variables from the same constraint graph from appearing on both sides. The test set contains a separate 40 instances.',[
    'results/split_manifest.csv, authoritative instance-to-partition mapping.',
    'student_starter.ipynb, cell 8 split recipe; MAAI801 Mini-Project Proposal, Section 3.6.'
  ]);
}
// 5. Models and selection
{
  const s=newSlide('Models and model choice',5);
  text(s,'Both models use the same six features, fit groups, and validation groups.',82,138,1090,42,{size:25,color:C.muted});
  makeTable(s,[['Model','Selected validation settings','Validation MAE'],
    ['Decision Tree',paramsLabel('Decision Tree'),f3(SUMMARY.dt_validation_mae)],
    ['Random Forest',paramsLabel('Random Forest'),f3(SUMMARY.rf_validation_mae)]],
    82,218,1110,230,[240,610,260]);
  text(s,'Selected family',82,503,230,34,{size:23,bold:true,color:C.muted});
  text(s,SUMMARY.selected_model,320,495,380,49,{size:34,bold:true,color:C.teal});
  text(s,'Lowest validation MAE; exact tie would favor Decision Tree.',708,504,480,36,{size:22,color:C.ink});
  text(s,'After locking the family and settings, each model was refit on all 160 training instances.',82,574,1080,45,{size:22,color:C.muted});
  notes(s,'Student 2','1:20','Decision Tree grid: max depth 3, 5, 8, or unbounded; minimum leaf size 1, 5, or 10. Random Forest grid: 100 or 200 trees, depth 5 or unbounded, and minimum leaf size 1 or 5. The final family is chosen on validation MAE alone. Here, RF has the slightly lower validation MAE. Both selected settings were refit on all training instances before test evaluation.',[
    'results/dt_validation.csv and results/rf_validation.csv, full validation grids.',
    'results/final_summary.json, locked configurations and selection basis.',
    'work/decision_tree_notes.md and work/random_forest_notes.md.'
  ]);
}
// 6. Prediction performance
{
  const s=newSlide('Prediction error on held-out candidates',6);
  text(s,'MAE is average absolute prediction error, in violations per candidate.',82,138,1090,42,{size:25,color:C.muted});
  addChart(s,'bar',{position:{left:112,top:207,width:1044,height:360},title:'Mean absolute error by model and split',
    titleTextStyle:{fontSize:22,fill:C.dark,bold:true},categories:['Decision Tree','Random Forest'],
    series:[{name:'Validation',values:[SUMMARY.dt_validation_mae,SUMMARY.rf_validation_mae],fill:C.coral},
      {name:'Test',values:[SUMMARY.dt_test_mae,SUMMARY.rf_test_mae],fill:C.teal}],
    hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:18,fill:C.muted}},
    barOptions:{direction:'column',grouping:'clustered',gapWidth:85},
    xAxis:{textStyle:{fontSize:19,fill:C.ink},line:{style:'solid',fill:C.line,width:1}},
    yAxis:{title:{text:'MAE (violations per candidate)',textStyle:{fontSize:17,fill:C.muted}},min:0,max:.6,majorUnit:.1,numberFormatCode:'0.00',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{style:'solid',fill:'#DDE3E1',width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:16,fill:C.ink}},chartFill:'none',plotAreaFill:'none'});
  text(s,'The Random Forest had the lowest validation MAE (0.515) and test MAE (0.437).',82,608,1100,37,{size:24,bold:true,color:C.dark});
  text(s,'Prediction error describes numeric accuracy; it does not measure variable-choice quality.',82,649,1100,28,{size:19,color:C.muted});
  notes(s,'Student 2','1:10','Read the four values precisely from the chart. Random Forest was selected on validation MAE alone, then refit on all 160 training instances. The test MAE is reported as final evaluation evidence; it did not drive model choice. Lower MAE is not equivalent to better state-level selection.',[
    'results/final_prediction_metrics.csv and results/final_summary.json.'
  ]);
}
// 7. Fair one-step comparison
{
  const s=newSlide('Same state, same mutation operator',7);
  text(s,'Each selector starts from an independent copy of the same original assignment.',82,140,1110,42,{size:25,color:C.muted});
  box(s,82,255,255,128,C.white,C.line,8);
  text(s,'Original state',106,277,210,34,{size:26,bold:true,align:'center'});
  text(s,'8 variables · one assignment',100,323,220,34,{size:21,color:C.muted,align:'center'});
  line(s,338,319,450,319,C.ink,3);
  line(s,450,319,450,245,C.ink,3); line(s,450,319,450,418,C.ink,3);
  line(s,450,245,520,245,C.coral,3); line(s,450,418,520,418,C.teal,3);
  box(s,520,181,300,128,C.coralPale,C.coral,8);
  text(s,'Percentage heuristic',544,205,250,34,{size:24,bold:true,color:C.coral,align:'center'});
  text(s,'Highest violation percentage',541,251,260,30,{size:19,align:'center'});
  box(s,520,354,300,128,C.tealPale,C.teal,8);
  text(s,'Random Forest',544,378,250,34,{size:25,bold:true,color:C.teal,align:'center'});
  text(s,'Highest predicted improvement',541,424,260,30,{size:19,align:'center'});
  line(s,820,245,891,245,C.ink,3); line(s,820,418,891,418,C.ink,3);
  line(s,891,245,891,319,C.ink,3); line(s,891,418,891,319,C.ink,3); line(s,891,319,948,319,C.ink,3);
  box(s,948,255,250,128,C.white,C.dark,8);
  text(s,'Same best-value\nmutation',970,277,207,63,{size:24,bold:true,align:'center'});
  text(s,'Compare actual gain',963,341,220,26,{size:18,color:C.muted,align:'center'});
  text(s,'Heuristic ties: percentage → incident violations → lower index · ML ties: lower index',82,560,1110,40,{size:20,color:C.muted});
  notes(s,'Student 3','1:20','For each test state, both methods begin from a fresh copy of the original assignment. The heuristic picks the variable with the highest percentage of incident violated constraints, with the supplied tie rules. The model predicts improvement for all eight variables and picks the largest predicted value. Both choices use the same deterministic best-value mutation. This isolates variable choice for the one-step comparison.',[
    'supplied/MAAI801_MaxCSP_Student_Package/maxcsp_tools.py, heuristic_select, best_value_mutation, evaluate_heuristic, evaluate_ml_selection.',
    'work/compare_selection.py, same-state comparison contract.'
  ]);
}
// 8. Actual outcomes
{
  const s=newSlide('Actual one-step results',8);
  const rel=SUMMARY.mean_gain_difference/SUMMARY.heuristic_mean_improvement;
  const winPct=SUMMARY.wins/SUMMARY.n_states, tiePct=SUMMARY.ties/SUMMARY.n_states, lossPct=SUMMARY.losses/SUMMARY.n_states;
  addChart(s,'bar',{position:{left:72,top:177,width:520,height:330},title:'Mean actual gain per state',
    titleTextStyle:{fontSize:19,fill:C.dark,bold:true},categories:['Heuristic','Random Forest'],
    series:[{name:'Violations reduced',values:[SUMMARY.heuristic_mean_improvement,SUMMARY.ml_mean_improvement],fill:C.teal,
      points:[{idx:0,fill:C.coral},{idx:1,fill:C.teal}]}],
    hasLegend:false,barOptions:{direction:'bar',grouping:'clustered',gapWidth:55},
    xAxis:{textStyle:{fontSize:17,fill:C.ink},line:{style:'solid',fill:C.line,width:1}},
    yAxis:{min:0,max:2,majorUnit:.5,numberFormatCode:'0.0',textStyle:{fontSize:15,fill:C.muted},majorGridlines:{style:'solid',fill:'#DDE3E1',width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:16,fill:C.ink}},chartFill:'none',plotAreaFill:'none'});
  makeTable(s,[['Mean outcome','Heuristic','Random Forest'],['Gain (violations reduced)',f3(SUMMARY.heuristic_mean_improvement),f3(SUMMARY.ml_mean_improvement)],
    ['Remaining violations',f3(SUMMARY.heuristic_mean_violations_after),f3(SUMMARY.ml_mean_violations_after)],['States evaluated',String(SUMMARY.n_states),String(SUMMARY.n_states)]],
    628,170,575,220,[270,150,155]);
  makeTable(s,[['ML vs heuristic','States','Share'],['Win',String(SUMMARY.wins),p2(winPct)],['Tie',String(SUMMARY.ties),p2(tiePct)],['Loss',String(SUMMARY.losses),p2(lossPct)]],
    628,418,575,192,[270,150,155]);
  text(s,`+${f3(SUMMARY.mean_gain_difference)} violations gained per state (${(rel*100).toFixed(1)}% relative)`,82,548,520,56,{size:25,bold:true,color:C.teal});
  text(s,`${SUMMARY.variable_disagreement_count}/${SUMMARY.n_states} states selected different variables`,82,615,520,30,{size:20,color:C.muted});
  notes(s,'Student 3','1:45',`On ${SUMMARY.n_states} held-out states, mean gain was ${SUMMARY.ml_mean_improvement.toFixed(3)} for Random Forest and ${SUMMARY.heuristic_mean_improvement.toFixed(3)} for the heuristic, a difference of ${SUMMARY.mean_gain_difference.toFixed(3)} violations per state (${(rel*100).toFixed(1)}% relative to the heuristic mean). ML won ${SUMMARY.wins}, tied ${SUMMARY.ties}, and lost ${SUMMARY.losses}; the selectors chose different variables in ${SUMMARY.variable_disagreement_count} states. The average difference is modest and the per-state results are mostly ties. State wins are counted on the supplied test states; instance-level outcomes are 18 wins, 20 ties, and 2 losses.`,[
    'results/final_summary.json, aggregate results.',
    'results/final_state_comparison.csv, state-level metrics and same-state checks.',
    'results/final_instance_comparison.csv, instance-level aggregation.'
  ]);
}
// 9. Conclusion and limits
{
  const s=newSlide('Conclusion and scope',9);
  text(s,'A modest average gain in this synthetic one-step setting',82,143,1100,45,{size:29,bold:true,color:C.teal});
  text(s,`Random Forest raised mean actual gain by ${f3(SUMMARY.mean_gain_difference)} violations per state: ${SUMMARY.ml_mean_improvement.toFixed(3)} vs ${SUMMARY.heuristic_mean_improvement.toFixed(3)}.`,
    82,207,1070,68,{size:26,color:C.ink});
  line(s,82,304,1188,304,C.line,1.5);
  text(s,'What this result covers',82,341,500,36,{size:25,bold:true,color:C.dark});
  text(s,'200 held-out states · 40 synthetic instances\nOne selected variable · one best-value mutation',82,389,500,83,{size:23,color:C.muted});
  text(s,'What it does not establish',650,341,500,36,{size:25,bold:true,color:C.dark});
  text(s,'A complete GA result, runtime advantage, or performance on the paper’s instances.',650,389,490,72,{size:23,color:C.muted});
  text(s,'Presentation roles',82,518,500,32,{size:22,bold:true,color:C.dark});
  text(s,'Student 1: context and data   ·   Student 2: validation and models\nStudent 3: selection comparison   ·   Student 4: conclusion and questions',82,558,1090,53,{size:19,color:C.muted});
  text(s,'Questions',82,633,340,36,{size:24,bold:true,color:C.coral});
  notes(s,'Student 4','1:15','Conclude narrowly: this evaluation supports a modest increase in average actual one-step gain for the selected Random Forest on these synthetic held-out states. There were 23 wins, 172 ties, and 5 losses across states. The result does not demonstrate a complete-GA improvement, speedup, transfer to the original paper’s instances, or generalization beyond this setting. Use the listed speaker roles, then invite questions. All presenters should be ready to explain the shared method.',[
    'results/final_summary.json and results/final_state_comparison.csv.',
    'MAAI801 Mini-Project Proposal, Sections 4.1–4.5 and scope statement.',
    'Ben Othman & Tagina (2015), DOI: https://doi.org/10.1016/j.procs.2015.08.156.'
  ]);
}

await fs.mkdir(STAGING,{recursive:true});
await fs.mkdir(path.dirname(FINAL_PPTX),{recursive:true});
for (const [i,slide] of presentation.slides.items.entries()) {
  const blob=await presentation.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(BUILD_DIR,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const expectedSlideSizeEmu = `${Math.round(W*9525)},${Math.round(H*9525)}`;
const draft=path.join(STAGING,'draft.pptx');
await (await PresentationFile.exportPptx(presentation)).save(draft);
const requirements={
  explicitTotalSlideCount:9,
  requiredNativeChartOwnerSlides:[3,6,8],
  requiredNativeTableOwnerSlides:[4,5,8],
  nativeChartTargetApplication:'powerpoint',
};
const result=await finalizePresentation({
  ...requirements,workspaceDir:ROOT,candidatePath:draft,finalPath:FINAL_PPTX,
  pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu',expectedSlideSizeEmu,'--validate-bullet-geometry','--validate-heading-fit',
    ...requirements.requiredNativeTableOwnerSlides.flatMap(n=>['--require-native-table-slide',String(n)])],
  requiredNativeTableOwnerSlides:requirements.requiredNativeTableOwnerSlides,
  fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
  receiptPath:path.join(STAGING,'presentation.validation.json'),
});
await fs.rm(path.join(ROOT,'submission/presentation.pptx'),{force:true});
await fs.copyFile(FINAL_PPTX,path.join(ROOT,'submission/presentation.pptx'));
console.log(JSON.stringify({result,finalPptx:path.join(ROOT,'submission/presentation.pptx'),validatedRevision:FINAL_PPTX,slideCount:presentation.slides.items.length,slideSize:{width:W,height:H}},null,2));
