"""Generate a reproducible synthetic supervised Max-CSP dataset and eval states.

python generate_dataset.py
"""
import csv
import json
import random
from pathlib import Path
from maxcsp_tools import make_instance, candidate_features, best_value_mutation

OUT = Path(__file__).resolve().parent
SEED = 20261028
FEATURES = ['variable_degree','variable_violations','template_percentage',
            'current_violations','current_satisfaction','num_constraints']


def main():
    rng = random.Random(SEED)
    instances, candidate_rows, state_rows = [], [], []
    # Different CSP structures per instance. Split is by instance (not row).
    for split, count in [('train', 160), ('test', 40)]:
        for ix in range(count):
            instance_id = f'{split}_{ix:03d}'
            instance = make_instance(instance_id, n_vars=8, domain_size=4,
                       density=rng.choice([0.25,0.45,0.65,0.85]),
                       tightness=rng.choice([0.2,0.4,0.6,0.8]), seed=rng.randrange(2**31))
            instances.append(instance)
            for state_ix in range(5):
                assignment = [rng.randrange(instance['domain_size']) for _ in range(instance['n_vars'])]
                state_id = f'{instance_id}_s{state_ix}'
                state_rows.append({'split':split,'instance_id':instance_id,
                                   'state_id':state_id,'assignment_json':json.dumps(assignment)})
                for j in range(instance['n_vars']):
                    base = candidate_features(instance, assignment, j)
                    outcome = best_value_mutation(instance, assignment, j)
                    candidate_rows.append({'split':split,'instance_id':instance_id,
                        'state_id':state_id,'variable_index':j,**base,
                        'improvement':outcome['improvement']})
    (OUT/'instances.json').write_text(json.dumps(instances,indent=1), encoding='utf-8')
    with (OUT/'candidate_dataset.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidate_rows[0]))
        writer.writeheader();writer.writerows(candidate_rows)
    with (OUT/'states.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(state_rows[0]))
        writer.writeheader();writer.writerows(state_rows)
    print(f'{len(instances)} instances, {len(state_rows)} states, {len(candidate_rows)} candidates')
    print('Train instances: 160, test instances: 40 (no shared instances)')

if __name__=='__main__':main()
