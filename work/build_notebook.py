from pathlib import Path
import ast,json,nbformat
P=Path(__file__).resolve().parents[1]
source=P/'supplied'/'MAAI801_MaxCSP_Student_Package'/'student_starter.ipynb'
nb=nbformat.read(source,as_version=4)
def getfun(file,name):
 s=(P/'work'/file).read_text();t=ast.parse(s)
 node=next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 return ast.get_source_segment(s,node)
nb.cells[0].source='''# Machine learning guided variable selection for Max CSP
## MAAI801 completed mini project
This notebook explores the supplied training data, tunes Decision Tree and Random Forest regressors using instance-grouped validation, locks a model, and compares one-step decisions on held-out instances. The data and constraint/mutation helpers are supplied by the course. AI assistance generated the analysis and draft explanations; the submitting team must review and understand them.

Run all cells in order. Keep the supplied data files beside this notebook, in `supplied/MAAI801_MaxCSP_Student_Package`, or upload the original student ZIP in Google Colab. Outputs are saved under `results/` and `figures/` in the current directory. No full genetic algorithm is implemented.'''
nb.cells[1].source='''from pathlib import Path
import sys, zipfile, os
OUTPUT = Path.cwd()
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.matplotlib_cache'))
candidates = [OUTPUT, OUTPUT/'MAAI801_MaxCSP_Student_Package', OUTPUT/'supplied'/'MAAI801_MaxCSP_Student_Package', OUTPUT.parent/'supplied'/'MAAI801_MaxCSP_Student_Package']
ROOT = next((p for p in candidates if (p/'candidate_dataset.csv').exists()), None)
if ROOT is None:
    for archive in OUTPUT.glob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            if any(n.endswith('candidate_dataset.csv') for n in z.namelist()):
                z.extractall(OUTPUT)
                break
    ROOT = next((p.parent for p in OUTPUT.rglob('candidate_dataset.csv')), None)
if ROOT is None:
    raise FileNotFoundError('Upload the original course student ZIP or place the supplied files beside this notebook.')
ROOT = ROOT.resolve()
sys.path.insert(0, str(ROOT))
for name in ['results','figures']:
    (OUTPUT/name).mkdir(exist_ok=True)
print('Data directory:', ROOT)
'''
nb.cells[2].source+='''\nfrom sklearn.model_selection import GroupShuffleSplit
train_df=data.loc[data['split'].eq('train')].copy()
test_df=data.loc[data['split'].eq('test')].copy()
splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
idx_tr,idx_val=next(splitter.split(train_df,groups=train_df.instance_id))
fit_df=train_df.iloc[idx_tr].copy();val_df=train_df.iloc[idx_val].copy()
fit_ids=set(fit_df.instance_id);val_ids=set(val_df.instance_id);test_ids=set(test_df.instance_id)
assert fit_ids.isdisjoint(val_ids) and fit_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
assert (len(fit_df),len(val_df),len(test_df))==(5120,1280,1600)
assert data.groupby('state_id').size().eq(8).all()
assert data.groupby('instance_id').state_id.nunique().eq(5).all()
assert not train_df[FEATURES+[TARGET]].isna().any().any()
assert train_df[TARGET].ge(0).all()
manifest=pd.DataFrame([(i,p) for ids,p in [(fit_ids,'fit'),(val_ids,'validation'),(test_ids,'test')] for i in sorted(ids)],columns=['instance_id','partition']).sort_values('instance_id')
manifest.to_csv(OUTPUT/'results'/'split_manifest.csv',index=False)
bundle={'data':data,'train_df':train_df,'fit_df':fit_df,'val_df':val_df,'test_df':test_df,'states':states,'instances':instances,'manifest':manifest}
print('Instance groups:',manifest.partition.value_counts().to_dict())
print('Missing training feature/target values:',int(train_df[FEATURES+[TARGET]].isna().sum().sum()))
print('Duplicate candidate keys:',int(train_df.duplicated(['instance_id','state_id','variable_index']).sum()))
'''
nb.cells[5].source='''## Task 1 Explore the training data
One row represents one candidate variable in a current assignment. The target is the reduction in violations after the supplied best-value change of that variable. All records from one instance share a constraint structure, so validation splits keep them together. IDs, split labels, target, and post-mutation outcomes are excluded from predictive inputs.'''
nb.cells[6].source="IDENTIFIERS=['instance_id','state_id','variable_index']\n"+getfun('data_exploration.py','training_eda')+'''\nsummary_eda=training_eda(bundle,OUTPUT)
display(train_df[FEATURES+[TARGET]].describe())
print('Training target counts:',summary_eda['target_counts'].to_dict())
print('Zero improvement percentage:',100*summary_eda['zero_target_rate'])
from IPython.display import Image,display
for filename in ['eda_target_distribution.png','eda_improvement_by_template_percentage.png']:
    display(Image(filename=str(OUTPUT/'figures'/filename)))
'''
nb.cells.insert(7,nbformat.v4.new_markdown_cell('''The target is concentrated at zero: 3,077 of 6,400 training candidates (48.1%) produce no strict improvement. Most positive gains are small. Percentage bands broadly track higher improvement, but the relationship is nonmonotonic, so percentage alone does not determine the best variable. Band means also depend on which candidates fall in each band and do not imply causation.'''))
# Original cells after insertion shift by one.
nb.cells[9].source='''from typing import Sequence
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
MAX_DEPTH_GRID=(3,5,8,None)
MIN_SAMPLES_LEAF_GRID=(1,5,10)
RANDOM_STATE=42
'''+getfun('decision_tree.py','train_and_validate').replace('def train_and_validate(','def tune_tree(')+'''\ndt_fit,dt_table,dt_params=tune_tree(fit_df,val_df,FEATURES,TARGET)
dt_table.to_csv(OUTPUT/'results'/'dt_validation.csv',index=False)
display(dt_table.sort_values('validation_mae'))
print('Selected Decision Tree:',dt_params)
'''
nb.cells[11].source='''from typing import Any
from sklearn.ensemble import RandomForestRegressor
'''+getfun('random_forest.py','train_and_validate').replace('def train_and_validate(','def tune_forest(')+'''\nrf_fit,rf_table,rf_params=tune_forest(fit_df,val_df,FEATURES,TARGET)
rf_table.to_csv(OUTPUT/'results'/'rf_validation.csv',index=False)
display(rf_table)
print('Selected Random Forest:',rf_params)
'''
nb.cells[12].source+='''\n\nThe search considers 12 tree settings and 8 forest settings at seed 42. Within-family exact ties prefer simpler configurations as encoded above. Select the family by validation MAE, preferring the Decision Tree on an exact tie. The Random Forest wins by a small validation margin of about 0.00350 violations. The margin alone does not demonstrate a robust advantage across new splits.'''
pipeline=(P/'work'/'run_final_evaluation.py').read_text()
eval_code=pipeline[pipeline.index('dt_val='):]
# environment imports and independent comparison helper supplied directly
nb.cells[13].source='''import platform,sklearn,matplotlib
PROJECT=OUTPUT
'''+getfun('compare_selection.py','compare_selection')+'\n'+eval_code+'''\ndisplay(metrics)
display(comparison.head())
display(byinstance)
for filename in ['prediction_mae.png','selection_comparison.png']:
    display(Image(filename=str(OUTPUT/'figures'/filename)))
'''
nb.cells[14].source='''## Task 5 Reflection

1. **Prediction accuracy.** The Random Forest has lower validation MAE (0.514774 versus 0.518270) and test MAE (0.436549 versus 0.443934). Family selection used validation only. Both selected configurations were refit on all training instances before test scoring, so validation and test MAEs also differ in training-set size.
2. **Different variables.** ML chooses a different variable in 40/200 states (20%). Twelve of these disagreements tie on actual gain; different selected indices need not mean different outcomes.
3. **Actual outcomes.** ML wins in 23 states (11.5%), ties in 172 (86%), and loses in 5 (2.5%). Its mean gain is 1.690 versus 1.565, a descriptive advantage of 0.125 violations per state (about 8.0%). Remaining violations average 5.940 versus 6.065 from a shared starting mean of 7.630. On mean gain per instance, ML improves 18, ties 20, and worsens 2 of 40 test instances. The per-instance table above identifies them.
4. **MAE and choices.** MAE averages numerical errors over all candidates, whereas selection uses their within-state ordering. A low error on unselected variables can coexist with a wrong top choice. The available aggregate features can also give the same inputs to variables whose alternative domain values have different effects.
5. **Scope.** This supports a modest observed advantage for this frozen one-step selector on these held-out synthetic states. There is no population, crossover, repeated trajectory, or runtime measurement, so the evidence does not establish faster or better performance for a complete genetic algorithm.
6. **Limitations.** Instances contain only eight variables and four values. There is one grouped validation split, a small model search, and only 40 held-out problem instances with five correlated states each. No significance claim or generalization to all Max-CSPs is made. Future work could repeat the protocol on new instances or study ranking objectives without reusing this test set for tuning.

## References
Ben Othman, H., & Tagina, M. (2015). Guided Genetic Algorithm: A New Template Concept. *Procedia Computer Science, 60*, 1101–1108. https://doi.org/10.1016/j.procs.2015.08.156. Publisher abstract: https://www.sciencedirect.com/science/article/pii/S1877050915022838.

MAAI801 course assignment and supplied Max-CSP student package (2026). The original paper motivates percentage guidance; this notebook evaluates the course’s simplified supervised extension.
'''
for c in nb.cells:
 if c.cell_type=='code':c.outputs=[];c.execution_count=None
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb,P/'submission'/'completed_student.ipynb')
print('Notebook created:',len(nb.cells),'cells')
