"""Training-only EDA and reusable, instance-grouped data loader for the Max-CSP project.

Run from the project root:
    python3 work/data_exploration.py

The returned test_df is reserved untouched for the coordinator's final evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [
    "variable_degree",
    "variable_violations",
    "template_percentage",
    "current_violations",
    "current_satisfaction",
    "num_constraints",
]
TARGET = "improvement"
IDENTIFIERS = ["instance_id", "state_id", "variable_index"]
REQUIRED_COLUMNS = ["split", *IDENTIFIERS, *FEATURES, TARGET]


def load_data(project_root: str | Path) -> dict[str, object]:
    """Load supplied data and create the shared grouped fit/validation split.

    Args:
        project_root: Folder containing candidate_dataset.csv, states.csv,
            instances.json, and maxcsp_tools.py.

    Returns a dict with keys ``data`` (raw candidate table), ``train_df``
    (provided train partition), ``fit_df``, ``val_df``, ``test_df`` (provided
    test partition, reserved for final evaluation), ``states``, ``instances``,
    ``manifest`` (one row per instance with partition), ``features``,
    ``target``, and ``project_root``. This is the shared interface for the
    model workers: ``bundle = load_data(root)`` then use ``bundle['fit_df']``,
    ``bundle['val_df']``, ``FEATURES`` and ``TARGET``. Do not use ``test_df``
    before final evaluation.
    """
    root = Path(project_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from maxcsp_tools import load_instances
    for name in ("candidate_dataset.csv", "states.csv", "instances.json", "maxcsp_tools.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required file missing: {root / name}")

    data = pd.read_csv(root / "candidate_dataset.csv")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Candidate data missing required columns: {missing_columns}")
    train_df = data.loc[data["split"].eq("train")].copy()
    test_df = data.loc[data["split"].eq("test")].copy()

    # Match the starter notebook's exact split recipe, grouping by instance.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    fit_idx, val_idx = next(splitter.split(train_df, groups=train_df["instance_id"]))
    fit_df = train_df.iloc[fit_idx].copy()
    val_df = train_df.iloc[val_idx].copy()

    fit_ids = set(fit_df["instance_id"])
    val_ids = set(val_df["instance_id"])
    test_ids = set(test_df["instance_id"])
    assert fit_ids.isdisjoint(val_ids)
    assert fit_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(fit_ids) == 128 and len(val_ids) == 32 and len(test_ids) == 40
    assert (len(fit_df), len(val_df), len(test_df)) == (5120, 1280, 1600)

    # Every state contributes all eight variable candidates to one partition.
    partition_by_instance = {**{i: "fit" for i in fit_ids},
                             **{i: "validation" for i in val_ids},
                             **{i: "test" for i in test_ids}}
    manifest = pd.DataFrame({"instance_id": sorted(partition_by_instance),
                             "partition": [partition_by_instance[i]
                                           for i in sorted(partition_by_instance)]})
    counts_by_state = data.groupby("state_id")["variable_index"].agg(["size", "nunique"])
    assert counts_by_state["size"].eq(8).all()
    assert counts_by_state["nunique"].eq(8).all()
    assert data.groupby("state_id")["instance_id"].nunique().eq(1).all()
    assert data.groupby("instance_id")["state_id"].nunique().eq(5).all()
    assert data.groupby("state_id")["split"].nunique().eq(1).all()
    assert train_df[TARGET].ge(0).all()
    assert not train_df[FEATURES].isna().any().any()

    states = pd.read_csv(root / "states.csv")
    instances = load_instances(root / "instances.json")
    assert len(states.loc[states["split"].eq("train")]) == 800
    assert len(states.loc[states["split"].eq("test")]) == 200
    assert set(states["instance_id"]) == set(data["instance_id"])
    return {"data": data, "train_df": train_df, "fit_df": fit_df,
            "val_df": val_df, "test_df": test_df, "states": states,
            "instances": instances, "manifest": manifest,
            "features": FEATURES.copy(), "target": TARGET,
            "project_root": root}


def training_eda(bundle: dict[str, object], output_root: str | Path) -> dict[str, object]:
    """Create EDA summaries and figures from the provided training rows only."""
    out = Path(output_root).resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    train = bundle["train_df"]
    # The provided held-out test partition is never read or summarized here.
    assert isinstance(train, pd.DataFrame)
    target_counts = train[TARGET].value_counts().sort_index()
    zero_rate = float(train[TARGET].eq(0).mean())
    train_features = train[FEATURES]
    ranges = pd.DataFrame({"min": train_features.min(), "max": train_features.max()})
    train_dup_keys = int(train.duplicated(IDENTIFIERS).sum())
    train_duplicate_rows = int(train.duplicated().sum())
    train_state_count = int(train["state_id"].nunique())

    # Figure 1: target distribution; counts are based only on train instances.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.bar(target_counts.index.astype(int), target_counts.values, color="#3977a8", width=0.72)
    ax.set(title="Training target distribution", xlabel="Actual improvement (violations reduced)",
           ylabel="Candidate rows")
    ax.set_xticks(target_counts.index.astype(int))
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(out / "figures" / "eda_target_distribution.png", dpi=180)
    plt.close(fig)

    # Figure 2: binned relationship between pre-mutation template percentage and target.
    bins = np.arange(0, 110, 10)
    labels = [f"{a}–{a+10}" for a in bins[:-1]]
    binned = train[["template_percentage", TARGET]].copy()
    binned["percentage_band"] = pd.cut(binned["template_percentage"], bins=bins,
                                        labels=labels, include_lowest=True, right=True)
    trend = binned.groupby("percentage_band", observed=False)[TARGET].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    present = trend["count"].gt(0)
    ax.bar(np.arange(len(trend))[present], trend.loc[present, "mean"],
           color="#d27a35", width=0.72)
    ax.set(title="Observed improvement by template-percentage band (training data)",
           xlabel="Template percentage band", ylabel="Mean actual improvement")
    ax.set_xticks(np.arange(len(trend)), trend.index.astype(str), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(out / "figures" / "eda_improvement_by_template_percentage.png", dpi=180)
    plt.close(fig)

    return {"target_counts": target_counts, "zero_target_rate": zero_rate,
            "feature_ranges": ranges, "duplicate_key_rows": train_dup_keys,
            "duplicate_full_rows": train_duplicate_rows,
            "train_state_count": train_state_count,
            "binned_target": trend}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "supplied" /
                                "MAAI801_MaxCSP_Student_Package")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    bundle = load_data(args.project_root)
    bundle["manifest"].to_csv(args.output_root / "results" / "split_manifest.csv", index=False)
    summary = training_eda(bundle, args.output_root)
    print("Grouped partitions:", bundle["manifest"]["partition"].value_counts().to_dict())
    print("Candidate rows (fit/validation/test, structural counts):",
          len(bundle["fit_df"]), len(bundle["val_df"]), len(bundle["test_df"]))
    print("Training target counts:", summary["target_counts"].to_dict())
    print("Training zero-target rate:", summary["zero_target_rate"])
    print("Training exact duplicate candidate keys:", summary["duplicate_key_rows"])
    print("Training exact duplicate rows:", summary["duplicate_full_rows"])
    print("Training feature ranges:\n", summary["feature_ranges"])
    print("Training improvement by template-percentage band:\n", summary["binned_target"])


if __name__ == "__main__":
    main()
