"""Fair, same-state comparison of the supplied heuristic and an ML selector.

This module deliberately delegates selection and mutation to the supplied
``maxcsp_tools`` helpers. It does not fit a model or decide which states may be
evaluated; the caller must pass the intended states (training/validation for
development, or the locked test set for the coordinator's final evaluation).
"""
from __future__ import annotations

import json
import pandas as pd

from maxcsp_tools import evaluate_heuristic, evaluate_ml_selection, n_violations


def compare_selection(states, instances, model, features):
    """Compare heuristic and ML one-step choices on the same input states.

    Parameters
    ----------
    states : pandas.DataFrame or iterable of mappings/records
        Rows need ``state_id``, ``instance_id``, and ``assignment_json``. The
        helper does not filter by ``split``; choose the allowed partition before
        calling it.
    instances : mapping
        ``instance_id -> instance`` as returned by ``load_instances``.
    model : fitted regressor
        Must implement ``predict(DataFrame)`` over the requested feature names.
    features : sequence[str]
        The six pre-mutation feature columns, in the same order used to fit the
        model. The target and identifiers must not be included.

    Returns
    -------
    pandas.DataFrame
        One row per input state, including shared starting violations,
        each method's selected variable, improvement, and remaining
        violations, the ML outcome (win/tie/loss versus heuristic improvement),
        and whether the methods selected different variables.

    Notes
    -----
    The comparison evaluates each method independently from a fresh copy of the
    same original assignment. It uses ``evaluate_heuristic`` and
    ``evaluate_ml_selection`` so tie-breaking and best-value mutation behavior
    match the supplied implementation exactly.
    """
    records = states.to_dict(orient="records") if isinstance(states, pd.DataFrame) else list(states)
    output = []
    for state in records:
        state_id = state["state_id"]
        instance_id = state["instance_id"]
        try:
            instance = instances[instance_id]
        except KeyError as exc:
            raise KeyError(f"No instance found for state {state_id!r}: {instance_id!r}") from exc

        raw_assignment = state["assignment_json"]
        assignment = json.loads(raw_assignment) if isinstance(raw_assignment, str) else list(raw_assignment)
        before = n_violations(instance, assignment)

        # Both supplied evaluators treat assignments as read-only and the
        # mutation helper copies them. Pass distinct copies to make the
        # same-starting-state contract explicit and robust to future changes.
        heuristic = evaluate_heuristic(instance, list(assignment))
        ml = evaluate_ml_selection(instance, list(assignment), model, features)

        if heuristic["violations_before"] != before or ml["violations_before"] != before:
            raise AssertionError(f"Selectors did not start from the original state {state_id!r}")
        h_gain = heuristic["improvement"]
        ml_gain = ml["improvement"]
        outcome = "win" if ml_gain > h_gain else "loss" if ml_gain < h_gain else "tie"
        output.append({
            "state_id": state_id,
            "instance_id": instance_id,
            "violations_before": before,
            "heuristic_selected_variable": heuristic["selected_variable"],
            "heuristic_improvement": h_gain,
            "heuristic_violations_after": heuristic["violations_after"],
            "ml_selected_variable": ml["selected_variable"],
            "ml_improvement": ml_gain,
            "ml_violations_after": ml["violations_after"],
            "ml_outcome": outcome,
            "variable_disagreement": (
                heuristic["selected_variable"] != ml["selected_variable"]
            ),
        })
    return pd.DataFrame(output, columns=[
        "state_id", "instance_id", "violations_before",
        "heuristic_selected_variable", "heuristic_improvement",
        "heuristic_violations_after", "ml_selected_variable",
        "ml_improvement", "ml_violations_after", "ml_outcome",
        "variable_disagreement",
    ])
