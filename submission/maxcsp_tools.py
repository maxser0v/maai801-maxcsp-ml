"""Mini-project Max-CSP tools. Binary constraints forbid specified value pairs.

Each candidate row proposes changing ONE variable in a complete assignment.
The deterministic best-value operator tests all legal domain values and retains
current value on ties. This is a classroom simplification, not a full GGA.
"""
from __future__ import annotations
import json
import random
from pathlib import Path


def make_instance(instance_id, n_vars=8, domain_size=4, density=0.45, tightness=0.4, seed=0):
    rng = random.Random(seed)
    constraints = []
    for u in range(n_vars):
        for v in range(u + 1, n_vars):
            if rng.random() < density:
                forbidden = [[a, b] for a in range(domain_size)
                             for b in range(domain_size) if rng.random() < tightness]
                constraints.append({'u': u, 'v': v, 'forbidden': forbidden})
    # Prevent empty graph; otherwise candidate feature denominators could all be zero.
    if not constraints:
        constraints.append({'u': 0, 'v': 1, 'forbidden': [[0, 0]]})
    return {'instance_id': instance_id, 'n_vars': n_vars, 'domain_size': domain_size,
            'density': density, 'tightness': tightness, 'constraints': constraints}


def violated_constraints(instance, assignment):
    """Return indices of violated binary constraints."""
    if len(assignment) != instance['n_vars']:
        raise ValueError('Assignment length does not match instance')
    d = instance['domain_size']
    if any(not isinstance(v, int) or not 0 <= v < d for v in assignment):
        raise ValueError('Assignment values must be integers in 0..domain_size-1')
    return [i for i, c in enumerate(instance['constraints'])
            if [assignment[c['u']], assignment[c['v']]] in c['forbidden']]


def n_violations(instance, assignment):
    return len(violated_constraints(instance, assignment))


def satisfaction(instance, assignment):
    return len(instance['constraints']) - n_violations(instance, assignment)


def candidate_features(instance, assignment, var_index):
    """Features available BEFORE evaluating a proposed mutation."""
    if not 0 <= var_index < instance['n_vars']:
        raise ValueError('Invalid variable index')
    involved = [i for i,c in enumerate(instance['constraints'])
                if var_index in (c['u'],c['v'])]
    violated = set(violated_constraints(instance, assignment))
    degree = len(involved)
    num_violated = sum(i in violated for i in involved)
    return {'variable_degree': degree,
            'variable_violations': num_violated,
            'template_percentage': round(100.0 * num_violated / degree, 6) if degree else 0.0,
            'current_violations': len(violated),
            'current_satisfaction': len(instance['constraints']) - len(violated),
            'num_constraints': len(instance['constraints'])}


def best_value_mutation(instance, assignment, var_index):
    """Evaluate all domain values; return best non-worsening one-variable move.

    Equal-scoring outcomes prefer retaining the current value; if an improvement
    exists, the smallest improving value is selected deterministically.
    """
    before = n_violations(instance, assignment)
    current = assignment[var_index]
    best_val, best_nvc = current, before
    for value in range(instance['domain_size']):
        if value == current:
            continue
        modified = list(assignment)
        modified[var_index] = value
        score = n_violations(instance, modified)
        if score < best_nvc:
            best_val, best_nvc = value, score
    revised = list(assignment)
    revised[var_index] = best_val
    return {'selected_variable': var_index, 'old_value': current, 'new_value': best_val,
            'violations_before': before, 'violations_after': best_nvc,
            'improvement': before - best_nvc,
            'satisfied_before': len(instance['constraints']) - before,
            'satisfied_after': len(instance['constraints']) - best_nvc,
            'new_assignment': revised}


def heuristic_select(instance, assignment):
    """Select highest percentage of violated constraints. Tie: most violations then lowest index."""
    candidates = [(i, candidate_features(instance, assignment, i)) for i in range(instance['n_vars'])]
    return max(candidates, key=lambda t: (t[1]['template_percentage'], t[1]['variable_violations'], -t[0]))[0]


def evaluate_heuristic(instance, assignment):
    """Result of original percentage-based selection plus best-value mutation."""
    return best_value_mutation(instance, assignment, heuristic_select(instance, assignment))


def evaluate_ml_selection(instance, assignment, model, feature_columns):
    """Evaluate a student's fitted scikit-learn-like regression model on this state.

    Predict improvements for every variable using only pre-mutation features;
    apply same best-value operator as the heuristic for fair comparison.
    """
    import pandas as pd  # Student ML dependencies are optional for baseline generation.
    inputs = pd.DataFrame([candidate_features(instance, assignment, j)
                           for j in range(instance['n_vars'])])
    scores = model.predict(inputs[list(feature_columns)])
    best_index = max(range(instance['n_vars']), key=lambda j: (float(scores[j]), -j))
    result = best_value_mutation(instance, assignment, best_index)
    result['predicted_improvement'] = float(scores[best_index])
    return result


def load_instances(path):
    with open(path, encoding='utf-8') as f:
        return {i['instance_id']: i for i in json.load(f)}
