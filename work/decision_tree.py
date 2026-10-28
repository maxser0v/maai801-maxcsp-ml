"""Grouped-validation tuning for the Max-CSP Decision Tree baseline.

The caller supplies the shared fit/validation partitions; this module never
loads or scores the reserved test partition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


MAX_DEPTH_GRID = (3, 5, 8, None)
MIN_SAMPLES_LEAF_GRID = (1, 5, 10)
RANDOM_STATE = 42


def train_and_validate(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: Sequence[str],
    target: str,
) -> tuple[DecisionTreeRegressor, pd.DataFrame, dict[str, int | None]]:
    """Tune the declared grid and return its selected fit, table, and params.

    Validation MAE is minimized. Exact ties prefer a shallower maximum depth
    (with ``None`` ranked as unbounded/most complex), then a larger leaf size.
    """
    missing = (set(features) | {target}) - set(fit_df.columns)
    missing |= (set(features) | {target}) - set(val_df.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")
    if fit_df.empty or val_df.empty:
        raise ValueError("Fit and validation partitions must both be non-empty")

    X_fit, y_fit = fit_df[list(features)], fit_df[target]
    X_val, y_val = val_df[list(features)], val_df[target]
    rows: list[dict[str, object]] = []
    for depth in MAX_DEPTH_GRID:
        for leaf_size in MIN_SAMPLES_LEAF_GRID:
            model = DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=leaf_size,
                random_state=RANDOM_STATE,
            )
            model.fit(X_fit, y_fit)
            mae = float(mean_absolute_error(y_val, model.predict(X_val)))
            rows.append({
                "max_depth": depth,
                "min_samples_leaf": leaf_size,
                "random_state": RANDOM_STATE,
                "validation_mae": mae,
            })

    table = pd.DataFrame(rows)
    # Stable explicit simplicity ordering for exact MAE ties.
    depth_rank = table["max_depth"].map(lambda depth: float("inf") if pd.isna(depth) else depth)
    order = table.assign(_depth_rank=depth_rank).sort_values(
        ["validation_mae", "_depth_rank", "min_samples_leaf"],
        ascending=[True, True, False], kind="mergesort",
    )
    selected = order.iloc[0]
    best_params: dict[str, int | None] = {
        "max_depth": None if pd.isna(selected["max_depth"]) else int(selected["max_depth"]),
        "min_samples_leaf": int(selected["min_samples_leaf"]),
        "random_state": RANDOM_STATE,
    }
    table["selected"] = (
        table["max_depth"].eq(best_params["max_depth"])
        & table["min_samples_leaf"].eq(best_params["min_samples_leaf"])
    )

    best_model = DecisionTreeRegressor(**best_params)
    best_model.fit(X_fit, y_fit)
    return best_model, table, best_params


def run_shared_validation(project_root: str | Path | None = None) -> tuple[DecisionTreeRegressor, pd.DataFrame, dict[str, int | None]]:
    """Convenience runner using the saved shared instance manifest.

    The manifest controls fit/validation membership. It is checked against the
    supplied candidate table before any model fitting.
    """
    import sys
    root = Path(project_root or Path(__file__).resolve().parents[1])
    work_dir = Path(__file__).resolve().parent
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))
    from data_exploration import load_data

    bundle = load_data(root / "supplied" / "MAAI801_MaxCSP_Student_Package")
    saved_manifest = pd.read_csv(root / "results" / "split_manifest.csv")
    generated_manifest = bundle["manifest"].sort_values("instance_id").reset_index(drop=True)
    saved_manifest = saved_manifest.sort_values("instance_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(saved_manifest, generated_manifest)
    return train_and_validate(
        bundle["fit_df"], bundle["val_df"], bundle["features"], bundle["target"]
    )


if __name__ == "__main__":
    model, results, params = run_shared_validation()
    output = Path(__file__).resolve().parents[1] / "results" / "dt_validation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Selected parameters: {params}")
    print(f"Selected validation MAE: {results.loc[results['selected'], 'validation_mae'].iloc[0]:.6f}")
    print(f"Saved: {output}")
