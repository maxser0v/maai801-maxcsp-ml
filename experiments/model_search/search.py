"""Optional model search. Uses only supplied training instances to choose models.
Run from repository root: python3 experiments/model_search/search.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/maai801-model-search-mpl')
from pathlib import Path
import sys,json,time,hashlib,platform
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib,sklearn
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'supplied/MAAI801_MaxCSP_Student_Package'))
FEATURES=['variable_degree','variable_violations','template_percentage','current_violations','current_satisfaction','num_constraints']

def build(spec):
 family=spec['family'];p=spec['params']
 if family=='Decision Tree':return DecisionTreeRegressor(**p,random_state=42)
 if family=='Random Forest':return RandomForestRegressor(**p,random_state=42,n_jobs=2)
 if family=='Extra Trees':return ExtraTreesRegressor(**p,random_state=42,n_jobs=2)
 if family=='Histogram Boosting':return HistGradientBoostingRegressor(**p,early_stopping=False,random_state=42)
 if family=='Gradient Boosting':return GradientBoostingRegressor(**p,random_state=42)
 if family=='SVR':return make_pipeline(StandardScaler(),SVR(**p))
 if family=='KNN':return make_pipeline(StandardScaler(),KNeighborsRegressor(**p,n_jobs=2))
 if family=='Ridge':return make_pipeline(StandardScaler(),Ridge(**p))
 raise ValueError(family)

def choose_rows(frame,predictions):
 """Predicted argmax; stable lowest variable index resolves equal scores."""
 t=frame[['state_id','instance_id','variable_index','improvement']].copy()
 t['prediction']=predictions
 return t.sort_values(['state_id','prediction','variable_index'],ascending=[True,False,True],kind='stable').drop_duplicates('state_id')

def specs():
 out=[]
 def add(f,p,label=None):out.append({'id':f"trial_{len(out):03d}",'family':f,'params':p,'label':label or f})
 add('Random Forest',{'n_estimators':200,'min_samples_leaf':5,'max_depth':None},'Original Random Forest')
 for depth,leaf in [(5,20),(8,20),(10,10),(None,20)]:add('Decision Tree',{'max_depth':depth,'min_samples_leaf':leaf})
 for leaf in [2,5,10,20]:
  for features in [0.7,1.0]:
   if leaf==5 and features==1.0:continue
   add('Random Forest',{'n_estimators':300,'min_samples_leaf':leaf,'max_features':features,'max_depth':None})
 for leaf in [2,5,10,20]:
  for features in [.7,1.0]:add('Extra Trees',{'n_estimators':300,'min_samples_leaf':leaf,'max_features':features})
 for loss in ['squared_error','poisson','absolute_error']:
  for leaves,minimum in [(7,30),(15,20),(31,20)]:
   add('Histogram Boosting',{'loss':loss,'max_iter':220,'learning_rate':.05,'max_leaf_nodes':leaves,'min_samples_leaf':minimum,'l2_regularization':2.0})
 for loss in ['squared_error','huber']:
  for depth in [2,3,4]:add('Gradient Boosting',{'loss':loss,'n_estimators':220,'learning_rate':.04,'max_depth':depth,'min_samples_leaf':15})
 for C in [1,10,50]:
  for eps in [.1,.3]:add('SVR',{'C':C,'epsilon':eps,'gamma':'scale'})
 for k in [15,40,100]:add('KNN',{'n_neighbors':k,'weights':'distance'})
 add('Ridge',{'alpha':10.0})
 return out

def main():
 trials=specs()
 protocol={'purpose':'Optional extension: improve actual one-step selection gain.',
 'inputs':'Only six supplied pre-mutation features; fit only original training instances.',
 'folds':4,'group':'instance_id','shuffle':True,'cv_seed':20261106,
 'primary_selection':['highest pooled out-of-fold actual mean selection gain','lowest out-of-fold MAE','lexical trial id'],
 'secondary_selection':['lowest out-of-fold MAE','highest out-of-fold actual mean selection gain','lexical trial id'],
 'test_policy':'Original 40-instance test already viewed previously; report as exploratory. Evaluate only two locked champions plus original baseline; never retune from test or fresh results.',
 'fresh_policy':{'seeds':[2026110601,2026110602],'instances_per_seed':250,'states_per_instance':5,'n_vars':8,'domain_size':4,'density':[.25,.45,.65,.85],'tightness':[.2,.4,.6,.8],'source':'Exact distribution and helpers from supplied generate_dataset.py; separate seed streams, no fresh training.'},
 'trials':trials,'sklearn_version':sklearn.__version__}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 data=pd.read_csv(ROOT/'submission/candidate_dataset.csv')
 train=data.loc[data.split.eq('train')].reset_index(drop=True)
 assert train.instance_id.nunique()==160 and len(train)==6400
 folds=list(GroupKFold(n_splits=4,shuffle=True,random_state=20261106).split(train,groups=train.instance_id))
 fold_rows=[]
 for fold,(a,b) in enumerate(folds):
  assert set(train.iloc[a].instance_id).isdisjoint(set(train.iloc[b].instance_id))
  for inst in sorted(train.iloc[b].instance_id.unique()):fold_rows.append({'instance_id':inst,'validation_fold':fold})
 pd.DataFrame(fold_rows).to_csv(OUT/'cv_instance_folds.csv',index=False)
 # Baseline rule uses only current features, with the supplied deterministic ties.
 h=train.sort_values(['state_id','template_percentage','variable_violations','variable_index'],ascending=[True,False,False,True],kind='stable').drop_duplicates('state_id')
 heuristic_mean=float(h.improvement.mean())
 rows=[];all_fold_rows=[]
 for trial in trials:
  t=time.monotonic();oof=np.full(len(train),np.nan)
  for fold,(a,b) in enumerate(folds):
   model=build(trial);model.fit(train.iloc[a][FEATURES],train.iloc[a].improvement)
   pred=model.predict(train.iloc[b][FEATURES]);oof[b]=pred
   selected=choose_rows(train.iloc[b],pred)
   all_fold_rows.append({'trial_id':trial['id'],'fold':fold,'mae':mean_absolute_error(train.iloc[b].improvement,pred),'mean_actual_gain':float(selected.improvement.mean())})
  assert np.isfinite(oof).all()
  selected=choose_rows(train,oof)
  row={'trial_id':trial['id'],'family':trial['family'],'label':trial['label'],'params':json.dumps(trial['params'],sort_keys=True),'cv_mae':float(mean_absolute_error(train.improvement,oof)),'cv_mean_actual_gain':float(selected.improvement.mean()),'heuristic_mean_actual_gain':heuristic_mean,'cv_relative_gain_percent':100*(float(selected.improvement.mean())/heuristic_mean-1),'elapsed_seconds':round(time.monotonic()-t,2)}
  rows.append(row);pd.DataFrame(rows).to_csv(OUT/'cv_leaderboard.csv',index=False)
  pd.DataFrame(all_fold_rows).to_csv(OUT/'cv_fold_metrics.csv',index=False)
  np.save(OUT/(trial['id']+'_oof.npy'),oof)
  print(f"{len(rows)}/{len(trials)} {trial['family']} MAE={row['cv_mae']:.5f} gain={row['cv_mean_actual_gain']:.5f} ({row['elapsed_seconds']}s)",flush=True)
 table=pd.DataFrame(rows)
 primary=table.sort_values(['cv_mean_actual_gain','cv_mae','trial_id'],ascending=[False,True,True]).iloc[0]
 secondary=table.sort_values(['cv_mae','cv_mean_actual_gain','trial_id'],ascending=[True,False,True]).iloc[0]
 lookup={s['id']:s for s in trials}
 lock={'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),'primary':lookup[primary.trial_id],'mae_champion':lookup[secondary.trial_id],'baseline':trials[0],'primary_cv':primary.to_dict(),'mae_champion_cv':secondary.to_dict(),'note':'Frozen using training-only grouped cross-validation before final extension evaluation.'}
 (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2))
 for name,spec in [('selection_champion',lock['primary']),('mae_champion',lock['mae_champion']),('original_forest',lock['baseline'])]:
  model=build(spec).fit(train[FEATURES],train.improvement);joblib.dump(model,OUT/(name+'.joblib'))
 print('LOCKED',json.dumps(lock,indent=2),flush=True)

if __name__=='__main__':main()
