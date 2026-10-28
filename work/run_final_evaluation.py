from pathlib import Path
import sys,json,platform
import pandas as pd,numpy as np,sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_exploration import load_data,FEATURES,TARGET
from decision_tree import train_and_validate as tune_tree
from random_forest import train_and_validate as tune_forest

PROJECT=Path(__file__).resolve().parents[1]
bundle=load_data(PROJECT/'supplied'/'MAAI801_MaxCSP_Student_Package')
from compare_selection import compare_selection
_,dt_table,dt_params=tune_tree(bundle['fit_df'],bundle['val_df'],FEATURES,TARGET)
_,rf_table,rf_params=tune_forest(bundle['fit_df'],bundle['val_df'],FEATURES,TARGET)
dt_val=float(dt_table.validation_mae.min());rf_val=float(rf_table.validation_mae.min())
selected='Decision Tree' if dt_val<=rf_val else 'Random Forest'
lock={'selection_basis':'validation MAE only; exact tie favors Decision Tree',
      'selected_model':selected,'dt_params':dt_params,'rf_params':rf_params,
      'dt_validation_mae':dt_val,'rf_validation_mae':rf_val,
      'refit_policy':'Refit both locked configurations on all 160 training instances before final test.'}
(PROJECT/'results'/'model_selection_lock.json').write_text(json.dumps(lock,indent=2))
print('MODEL LOCKED:',json.dumps(lock),flush=True)
train_df=bundle['train_df'];test_df=bundle['test_df']
dt_model=DecisionTreeRegressor(**dt_params).fit(train_df[FEATURES],train_df[TARGET])
rf_model=RandomForestRegressor(**rf_params,n_jobs=-1).fit(train_df[FEATURES],train_df[TARGET])
models={'Decision Tree':dt_model,'Random Forest':rf_model};best_model=models[selected]
metrics=pd.DataFrame([{'model':name,'validation_mae':val,'test_mae':mean_absolute_error(test_df[TARGET],model.predict(test_df[FEATURES]))} for name,val,model in [('Decision Tree',dt_val,dt_model),('Random Forest',rf_val,rf_model)]])
test_states=bundle['states'].query("split == 'test'")
comparison=compare_selection(test_states,bundle['instances'],best_model,FEATURES)
assert len(comparison)==200 and comparison.state_id.is_unique
for prefix in ['heuristic','ml']:
 assert (comparison[f'{prefix}_improvement']==comparison.violations_before-comparison[f'{prefix}_violations_after']).all()
 assert comparison[f'{prefix}_improvement'].ge(0).all()
metrics.to_csv(PROJECT/'results'/'final_prediction_metrics.csv',index=False)
comparison.to_csv(PROJECT/'results'/'final_state_comparison.csv',index=False)
byinstance=comparison.groupby('instance_id')[['heuristic_improvement','ml_improvement']].mean()
byinstance['gain_difference']=byinstance.ml_improvement-byinstance.heuristic_improvement
byinstance.to_csv(PROJECT/'results'/'final_instance_comparison.csv')
counts=comparison.ml_outcome.value_counts().reindex(['win','tie','loss'],fill_value=0)
summary={**lock,'dt_test_mae':float(metrics.iloc[0].test_mae),'rf_test_mae':float(metrics.iloc[1].test_mae),'n_states':len(comparison),'n_test_candidates':len(test_df),'n_test_instances':40,'mean_starting_violations':float(comparison.violations_before.mean()),'heuristic_mean_improvement':float(comparison.heuristic_improvement.mean()),'ml_mean_improvement':float(comparison.ml_improvement.mean()),'heuristic_mean_violations_after':float(comparison.heuristic_violations_after.mean()),'ml_mean_violations_after':float(comparison.ml_violations_after.mean()),'wins':int(counts['win']),'ties':int(counts['tie']),'losses':int(counts['loss']),'variable_disagreement_count':int(comparison.variable_disagreement.sum()),'mean_gain_difference':float((comparison.ml_improvement-comparison.heuristic_improvement).mean()),'instance_wins':int(byinstance.gain_difference.gt(0).sum()),'instance_ties':int(byinstance.gain_difference.eq(0).sum()),'instance_losses':int(byinstance.gain_difference.lt(0).sum()),'environment':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
assert summary['wins']+summary['ties']+summary['losses']==200
(PROJECT/'results'/'final_summary.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained');x=np.arange(2);w=.34
ax.bar(x-w/2,metrics.validation_mae,w,label='Validation',color='#3977a8');ax.bar(x+w/2,metrics.test_mae,w,label='Test after refit',color='#d27a35')
ax.set_xticks(x,metrics.model);ax.set_ylabel('Mean absolute error (violations)');ax.set_title('Candidate prediction error');ax.legend();ax.set_ylim(0,.7);ax.grid(axis='y',alpha=.2)
fig.savefig(PROJECT/'figures'/'prediction_mae.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
axs[0].bar(['Heuristic',selected],[summary['heuristic_mean_improvement'],summary['ml_mean_improvement']],color=['#8a949e','#3977a8'])
axs[0].set_ylabel('Mean actual reduction in violations');axs[0].set_title('Actual improvement on 200 test states')
axs[1].bar(['ML wins','Ties','ML losses'],counts.values,color=['#3977a8','#8a949e','#d27a35']);axs[1].set_ylabel('Test states');axs[1].set_title('Paired decision outcomes')
for ax in axs: ax.grid(axis='y',alpha=.2)
fig.savefig(PROJECT/'figures'/'selection_comparison.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))
