"""Evaluate only frozen extension champions, without further model selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/maai801-model-search-mpl')
from pathlib import Path
import json,sys,random,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
from search import ROOT,OUT,FEATURES,choose_rows
from maxcsp_tools import make_instance,candidate_features,best_value_mutation,heuristic_select,load_instances,evaluate_ml_selection

def generate_fresh(protocol):
 rng_instances=[];candidate_rows=[];states=[]
 for seed in protocol['seeds']:
  rng=random.Random(seed)
  for ix in range(protocol['instances_per_seed']):
   iid=f'fresh_{seed}_{ix:03d}'
   inst=make_instance(iid,n_vars=8,domain_size=4,density=rng.choice(protocol['density']),tightness=rng.choice(protocol['tightness']),seed=rng.randrange(2**31))
   rng_instances.append(inst)
   for k in range(5):
    assignment=[rng.randrange(4) for _ in range(8)];sid=f'{iid}_s{k}'
    states.append({'instance_id':iid,'state_id':sid,'assignment_json':json.dumps(assignment),'cohort_seed':seed})
    for j in range(8):
     candidate_rows.append({'instance_id':iid,'state_id':sid,'variable_index':j,**candidate_features(inst,assignment,j),'improvement':best_value_mutation(inst,assignment,j)['improvement']})
 return pd.DataFrame(candidate_rows),pd.DataFrame(states),{i['instance_id']:i for i in rng_instances}

def cluster_interval(frame,column,seed=42):
 diff=frame.groupby('instance_id')[column].mean().to_numpy()
 rng=np.random.default_rng(seed)
 boot=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(axis=1)
 return [float(x) for x in np.quantile(boot,[.025,.975])]

def evaluate(name,frame,states,instances,models):
 frame=frame.reset_index(drop=True)
 base=frame.sort_values(['state_id','template_percentage','variable_violations','variable_index'],ascending=[True,False,False,True],kind='stable').drop_duplicates('state_id').set_index('state_id')
 result=states[['state_id','instance_id']].copy().set_index('state_id')
 result['heuristic_variable']=base.variable_index;result['heuristic_gain']=base.improvement
 metrics=[]
 for model_name,model in models.items():
  predictions=model.predict(frame[FEATURES]);selected=choose_rows(frame,predictions).set_index('state_id')
  result[model_name+'_variable']=selected.variable_index
  result[model_name+'_gain']=selected.improvement
  result[model_name+'_minus_heuristic']=result[model_name+'_gain']-result.heuristic_gain
  result[model_name+'_minus_original']=result[model_name+'_gain']-result.original_forest_gain if 'original_forest_gain' in result else 0
  delta=result[model_name+'_minus_heuristic']
  metrics.append({'dataset':name,'model':model_name,'states':len(result),'instances':result.instance_id.nunique(),'mae':float(mean_absolute_error(frame.improvement,predictions)),'mean_actual_gain':float(selected.improvement.mean()),'heuristic_mean_gain':float(result.heuristic_gain.mean()),'relative_gain_percent':100*(float(selected.improvement.mean())/float(result.heuristic_gain.mean())-1),'wins':int(delta.gt(0).sum()),'ties':int(delta.eq(0).sum()),'losses':int(delta.lt(0).sum()),'mean_gain_above_heuristic':float(delta.mean()),'gain_difference_cluster_95ci':cluster_interval(result,model_name+'_minus_heuristic')})
 # Reconcile every chosen outcome with the supplied mutation helper, independent of CSV target.
 for row in states.itertuples(index=False):
  assignment=json.loads(row.assignment_json);inst=instances[row.instance_id]
  assert heuristic_select(inst,assignment)==int(result.loc[row.state_id,'heuristic_variable'])
  for method in ['heuristic',*models]:
   chosen=int(result.loc[row.state_id,method+'_variable']);out=best_value_mutation(inst,list(assignment),chosen)
   assert out['improvement']==int(result.loc[row.state_id,method+'_gain'])
 # Exact supplied ML helper also verifies prediction order and ties on sample states.
 for row in states.iloc[:10].itertuples(index=False):
  for key,model in models.items():
   out=evaluate_ml_selection(instances[row.instance_id],json.loads(row.assignment_json),model,FEATURES)
   assert out['selected_variable']==int(result.loc[row.state_id,key+'_variable'])
 for metric in metrics:
  key=metric['model'];metric['mean_gain_above_original_forest']=float(result[key+'_minus_original'].mean())
  metric['gain_above_original_cluster_95ci']=cluster_interval(result,key+'_minus_original')
 result.to_csv(OUT/(name+'_state_results.csv'))
 return metrics,result

def main():
 lock=json.loads((OUT/'selection_lock.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
 assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
 # Model order ensures paired differences always refer to the original fitted forest.
 models={name:joblib.load(OUT/(name+'.joblib')) for name in ['original_forest','selection_champion','mae_champion']}
 frame=pd.read_csv(ROOT/'submission/candidate_dataset.csv').query("split == 'test'").copy()
 states=pd.read_csv(ROOT/'submission/states.csv').query("split == 'test'").copy()
 inst=load_instances(ROOT/'submission/instances.json')
 rows,original=evaluate('original_test_exploratory',frame,states,inst,models)
 original_summary=json.loads((ROOT/'results/final_summary.json').read_text())
 assert abs(original.original_forest_gain.mean()-original_summary['ml_mean_improvement'])<1e-12
 fresh,fs,fi=generate_fresh(protocol['fresh_policy'])
 fresh.to_csv(OUT/'fresh_candidates.csv',index=False);fs.to_csv(OUT/'fresh_states.csv',index=False)
 (OUT/'fresh_instances.json').write_text(json.dumps(list(fi.values())))
 fresh_rows,fresh_result=evaluate('fresh_test',fresh,fs,fi,models);rows+=fresh_rows
 for seed in protocol['fresh_policy']['seeds']:
  ids=set(fs.loc[fs.cohort_seed.eq(seed),'state_id'])
  subset=fresh_result.loc[fresh_result.index.isin(ids)]
  for model in models:
   gain=float(subset[model+'_gain'].mean());h=float(subset.heuristic_gain.mean())
   rows.append({'dataset':f'fresh_seed_{seed}','model':model,'states':len(subset),'instances':subset.instance_id.nunique(),'mean_actual_gain':gain,'heuristic_mean_gain':h,'relative_gain_percent':100*(gain/h-1),'mean_gain_above_original_forest':float(subset[model+'_minus_original'].mean())})
 table=pd.DataFrame(rows);table.to_csv(OUT/'evaluation_summary.csv',index=False)
 (OUT/'evaluation_summary.json').write_text(json.dumps(rows,indent=2))
 print(table[['dataset','model','mae','mean_actual_gain','heuristic_mean_gain','relative_gain_percent','mean_gain_above_original_forest']].to_string(index=False),flush=True)

if __name__=='__main__':main()
