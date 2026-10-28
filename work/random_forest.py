"""Grouped validation tuning for the Max-CSP Random Forest model.

Run from the project root with:
    python3 work/random_forest.py

Only the shared fit and validation partitions are used for model selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_and_validate(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[RandomForestRegressor, pd.DataFrame, dict[str, Any]]:
    """Tune the declared modest grid and return fitted winner, scores, params.

    Exact MAE ties prefer fewer trees, a shallower tree, then larger leaves.
    ``max_depth=None`` is treated as deeper than every finite candidate.
    """
    grid = [
        {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf}
        for n in (100, 200)
        for depth in (5, None)
        for leaf in (1, 5)
    ]
    records: list[dict[str, Any]] = []
    for params in grid:
        candidate = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        candidate.fit(fit_df[features], fit_df[target])
        predictions = candidate.predict(val_df[features])
        mae = float(mean_absolute_error(val_df[target], predictions))
        records.append({**params, "random_state": 42, "validation_mae": mae})

    validation = pd.DataFrame(records)
    # Stable and explicit deterministic tie resolution following the assignment.
    def tie_key(row: pd.Series) -> tuple[float, int, int, int]:
        depth = row["max_depth"]
        return (
            float(row["validation_mae"]),
            int(row["n_estimators"]),
            int(depth) if pd.notna(depth) else 10**9,
            -int(row["min_samples_leaf"]),
        )

    best_row = min((row for _, row in validation.iterrows()), key=tie_key)
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": int(best_row["max_depth"]) if pd.notna(best_row["max_depth"]) else None,
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "random_state": 42,
    }
    model = RandomForestRegressor(**best_params, n_jobs=-1)
    model.fit(fit_df[features], fit_df[target])
    validation = validation.sort_values(
        ["validation_mae", "n_estimators", "max_depth", "min_samples_leaf"],
        ascending=[True, True, True, False], na_position="last",
    ).reset_index(drop=True)
    return model, validation, best_params


def main() -> None:
    from data_exploration import load_data

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "supplied" / "MAAI801_MaxCSP_Student_Package"
    bundle = load_data(source_root)
    model, validation, best_params = train_and_validate(
        bundle["fit_df"], bundle["val_df"], bundle["features"], bundle["target"]
    )
    output = project_root / "results" / "rf_validation.csv"
    validation.to_csv(output, index=False)
    import json
    (project_root / "results" / "rf_best_params.json").write_text(
        json.dumps(best_params, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Selected params: {best_params}")
    print(f"Best validation MAE: {validation.iloc[0]['validation_mae']:.6f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
