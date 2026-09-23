from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech_annulus_experiments as exp


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("mech_multidataset_requested_sensitivity_dense")

SPLIT = {
    "n_train": 2800,
    "n_index": 1800,
    "n_queries": 40,
}

BASE = {
    "hamming_probe": 2,
    "min_collisions": 2,
    "annulus_count": 1,
    "ring_sample_size": 16,
    "bandwidth_factor": 1.0,
    "L": 12,
    "K": 8,
}

L_VALUES = list(range(2, 13))
K_VALUES = list(range(2, 13))
DELTA_FACTORS = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50]
RADIUS_PERCENTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]
BANDWIDTH_FACTORS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00]

FACTOR_LABELS = {
    "L": "Hash tables L",
    "K": "Hash code length K",
    "delta": "Sphere interval delta",
    "fixed_query_radius": "Annulus/query radius R",
    "bandwidth_factor": "Kernel bandwidth factor sigma",
}


def split_sizes(n_rows: int) -> dict[str, int]:
    if n_rows >= SPLIT["n_train"] + SPLIT["n_index"] + SPLIT["n_queries"]:
        return dict(SPLIT)

    n_queries = min(SPLIT["n_queries"], max(10, n_rows // 20))
    n_train = min(SPLIT["n_train"], max(100, int(n_rows * 0.45)))
    n_index = min(SPLIT["n_index"], n_rows - n_train - n_queries)
    if n_index < 100:
        n_index = max(1, n_rows - n_queries - 100)
        n_train = n_rows - n_queries - n_index
    if n_train <= 0 or n_index <= 0 or n_queries <= 0:
        raise ValueError(f"Cannot split dataset with {n_rows} rows.")
    return {"n_train": int(n_train), "n_index": int(n_index), "n_queries": int(n_queries)}


def format_cell(column: str, value) -> str:
    if column in {"dataset", "factor", "factor_label", "factor_value", "feasible"}:
        return str(value)
    if column in {
        "L",
        "K",
        "hamming_probe",
        "min_collisions",
        "annulus_count",
        "ring_sample_size",
        "radius_percentile",
    }:
        return str(int(value))
    if column in {"delta", "fixed_query_radius", "bandwidth_factor"}:
        return f"{float(value):.4f}"
    if column in {
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "candidate_expansion_ratio",
        "score_highR",
        "score_balanced",
    }:
        return f"{float(value):.4f}"
    return f"{float(value):.3f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_cell(col, row[col]) for col in columns) + " |")
    return "\n".join(lines)


def dataset_radius_values(distances: np.ndarray) -> tuple[dict[int, float], float]:
    values = {}
    for percentile in RADIUS_PERCENTILES:
        per_query = np.percentile(distances, percentile, axis=1)
        values[percentile] = float(np.median(per_query))
    return values, values[35]


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kde_quality"] = (1.0 - df["kde_abs_relative_error"]).clip(lower=0.0, upper=1.0)
    time_max = max(float(df["query_time_ms"].max()), 1e-12)
    df["speed_quality"] = (1.0 - df["query_time_ms"] / time_max).clip(lower=0.0, upper=1.0)
    df["score_highR"] = (
        0.40 * df["point_f1"]
        + 0.35 * df["point_recall"]
        + 0.15 * df["kde_quality"]
        + 0.10 * df["speed_quality"]
    )
    df["score_balanced"] = (
        0.45 * df["point_f1"]
        + 0.25 * df["point_recall"]
        + 0.20 * df["kde_quality"]
        + 0.10 * df["speed_quality"]
    )
    df["feasible"] = (df["point_recall"] >= 0.80) & (df["point_f1"] >= 0.60)
    return df


def best_by_factor(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTOR_LABELS:
        part = df[df["factor"] == factor].copy()
        feasible = part[part["feasible"]]
        if not feasible.empty:
            best = feasible.sort_values(
                ["score_highR", "point_f1", "query_time_ms"],
                ascending=[False, False, True],
            ).iloc[0]
        else:
            best = part.sort_values(
                ["score_highR", "point_recall", "point_f1"],
                ascending=[False, False, False],
            ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows)


def plot_dataset_curves(dataset: str, df: pd.DataFrame, out_dir: Path) -> None:
    metric_specs = [
        ("point_recall", "Recall R"),
        ("point_precision", "Precision P"),
        ("point_f1", "F1"),
        ("kde_abs_relative_error", "KDE Err."),
        ("query_time_ms", "Online ms"),
    ]
    plot_dir = out_dir / "figures" / dataset
    plot_dir.mkdir(parents=True, exist_ok=True)
    for factor, label in FACTOR_LABELS.items():
        part = df[df["factor"] == factor].sort_values("factor_value")
        x = part["factor_value"].astype(float).to_numpy()
        fig, axes = plt.subplots(1, len(metric_specs), figsize=(18, 3.4))
        fig.suptitle(f"{dataset}: {label}", x=0.01, ha="left", fontsize=12, fontweight="bold")
        for ax, (metric, title) in zip(axes, metric_specs):
            ax.plot(x, part[metric].astype(float).to_numpy(), marker="o", linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("value")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{factor}_sensitivity.png", dpi=220)
        plt.close(fig)


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    exp.set_seed(exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    x = exp.load_dataset(dataset)
    sizes = split_sizes(len(x))
    train_x, index_x, queries = exp.split_data(
        x,
        sizes["n_train"],
        sizes["n_index"],
        sizes["n_queries"],
        rng,
    )
    distances = exp.distance_matrix(index_x, queries)
    inner_base = np.percentile(distances, 15.0, axis=1)
    outer_base = np.percentile(distances, 25.0, axis=1)
    bandwidth_base = float(np.median(outer_base))
    radius_values, base_radius = dataset_radius_values(distances)
    base_delta = 0.25 * base_radius
    delta_values = [max(base_radius * factor, exp.EPS) for factor in DELTA_FACTORS]

    model = exp.train_mech(
        train_x,
        12,
        12,
        24,
        128,
        1e-3,
        1.0,
        0.1,
        0.1,
        0.1,
        "cpu",
    )

    base = {
        **BASE,
        "delta": base_delta,
        "fixed_query_radius": base_radius,
    }

    factors = {
        "L": L_VALUES,
        "K": K_VALUES,
        "delta": delta_values,
        "fixed_query_radius": [radius_values[p] for p in RADIUS_PERCENTILES],
        "bandwidth_factor": BANDWIDTH_FACTORS,
    }

    radius_to_percentile = {
        round(value, 10): percentile for percentile, value in radius_values.items()
    }

    index_cache = {}

    def get_index(tables: int, bits: int, hamming_probe: int):
        key = (int(tables), int(bits), int(hamming_probe))
        if key not in index_cache:
            index_cache[key] = exp.MECHHashIndex(model, index_x, *key)
        return index_cache[key]

    rows = []
    total = sum(len(values) for values in factors.values())
    done = 0
    for factor, values in factors.items():
        for value in values:
            setting = dict(base)
            setting[factor] = value
            index = get_index(setting["L"], setting["K"], setting["hamming_probe"])
            config = exp.EvalConfig(
                delta=float(setting["delta"]),
                radius_factor=1.0,
                bandwidth_factor=float(setting["bandwidth_factor"]),
                variant="full",
                query_mode="radius_first",
                min_collisions=int(setting["min_collisions"]),
                fixed_query_radius=float(setting["fixed_query_radius"]),
                annulus_count=int(setting["annulus_count"]),
                ring_sample_size=int(setting["ring_sample_size"]),
            )
            row = exp.evaluate_index(
                index,
                index_x,
                queries,
                distances,
                inner_base,
                outer_base,
                bandwidth_base,
                config,
            )
            row.update(setting)
            row["dataset"] = dataset
            row["factor"] = factor
            row["factor_label"] = FACTOR_LABELS[factor]
            row["factor_value"] = value
            row["radius_percentile"] = (
                radius_to_percentile.get(round(float(value), 10), np.nan)
                if factor == "fixed_query_radius"
                else np.nan
            )
            rows.append(row)
            done += 1
            print(
                f"[{dataset} {done:02d}/{total}] {factor}={value:.6g}: "
                f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                f"F1={row['point_f1']:.3f} KDE={row['kde_abs_relative_error']:.3f} "
                f"CER={row['candidate_expansion_ratio']:.3f} T={row['query_time_ms']:.3f}",
                flush=True,
            )

    df = score_rows(pd.DataFrame(rows))
    best = best_by_factor(df)

    final_params = dict(base)
    for _, row in best.iterrows():
        final_params[row["factor"]] = row["factor_value"]
    final_params["L"] = int(final_params["L"])
    final_params["K"] = int(final_params["K"])
    final_params["hamming_probe"] = int(final_params["hamming_probe"])
    final_params["min_collisions"] = int(final_params["min_collisions"])
    final_params["annulus_count"] = int(final_params["annulus_count"])
    final_params["ring_sample_size"] = int(final_params["ring_sample_size"])
    final_params["delta"] = float(final_params["delta"])
    final_params["fixed_query_radius"] = float(final_params["fixed_query_radius"])
    final_params["bandwidth_factor"] = float(final_params["bandwidth_factor"])

    final_index = get_index(final_params["L"], final_params["K"], final_params["hamming_probe"])
    final_config = exp.EvalConfig(
        delta=final_params["delta"],
        radius_factor=1.0,
        bandwidth_factor=final_params["bandwidth_factor"],
        variant="full",
        query_mode="radius_first",
        min_collisions=final_params["min_collisions"],
        fixed_query_radius=final_params["fixed_query_radius"],
        annulus_count=final_params["annulus_count"],
        ring_sample_size=final_params["ring_sample_size"],
    )
    final_row = exp.evaluate_index(
        final_index,
        index_x,
        queries,
        distances,
        inner_base,
        outer_base,
        bandwidth_base,
        final_config,
    )
    final_row.update(final_params)
    final_row["dataset"] = dataset
    final_df = pd.DataFrame([final_row])
    meta = {
        "dataset": dataset,
        "n_train": len(train_x),
        "n_index": len(index_x),
        "n_queries": len(queries),
        "requested_split": sizes,
        "bandwidth_base": bandwidth_base,
        "base_radius_percentile": 35,
        "base_radius": base_radius,
        "base_delta": base_delta,
        "radius_values": radius_values,
        "delta_values": delta_values,
        "final_params": final_params,
    }
    return df, best, final_df, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_best = []
    all_final = []
    metas = []

    for dataset in DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result_path = dataset_dir / "requested_parameter_sensitivity_dense.csv"
        best_path = dataset_dir / "best_by_factor_dense.csv"
        final_path = dataset_dir / "final_combined_evaluation.csv"
        if result_path.exists() and best_path.exists() and final_path.exists():
            print(f"Using existing results for {dataset}", flush=True)
            df = pd.read_csv(result_path)
            best = pd.read_csv(best_path)
            final_df = pd.read_csv(final_path)
            meta = {"dataset": dataset, "loaded_existing": True}
        else:
            df, best, final_df, meta = evaluate_dataset(dataset)
            df.to_csv(result_path, index=False)
            best.to_csv(best_path, index=False)
            final_df.to_csv(final_path, index=False)
        plot_dataset_curves(dataset, df, OUT_DIR)
        all_rows.append(df)
        all_best.append(best)
        all_final.append(final_df)
        metas.append(meta)

    all_results = pd.concat(all_rows, ignore_index=True)
    best_results = pd.concat(all_best, ignore_index=True)
    final_results = pd.concat(all_final, ignore_index=True)
    all_results.to_csv(OUT_DIR / "all_requested_parameter_sensitivity_dense.csv", index=False)
    best_results.to_csv(OUT_DIR / "best_by_dataset_factor.csv", index=False)
    final_results.to_csv(OUT_DIR / "final_combined_by_dataset.csv", index=False)
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "split": SPLIT,
                "base": BASE,
                "L_values": L_VALUES,
                "K_values": K_VALUES,
                "delta_factors": DELTA_FACTORS,
                "radius_percentiles": RADIUS_PERCENTILES,
                "bandwidth_factors": BANDWIDTH_FACTORS,
                "factor_labels": FACTOR_LABELS,
                "selection_rule": "Feasible if recall>=0.80 and F1>=0.60; choose max highR score among feasible settings.",
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    final_cols = [
        "dataset",
        "L",
        "K",
        "delta",
        "fixed_query_radius",
        "bandwidth_factor",
        "hamming_probe",
        "min_collisions",
        "annulus_count",
        "ring_sample_size",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "candidate_expansion_ratio",
        "candidate_size",
        "kde_sample_size",
        "filter_time_ms",
        "kde_time_ms",
        "query_time_ms",
    ]
    best_cols = [
        "dataset",
        "factor_label",
        "factor_value",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "candidate_expansion_ratio",
        "candidate_size",
        "kde_sample_size",
        "filter_time_ms",
        "kde_time_ms",
        "query_time_ms",
        "score_highR",
        "feasible",
    ]
    summary = [
        "# Multi-Dataset Requested Parameter Sensitivity",
        "",
        "## Setup",
        "",
        "- Datasets: isolet, cifar10, cifar10_gist512, amazon.",
        "- Each requested parameter uses 11-12 values and changes one factor at a time.",
        "- Radius values are dataset-specific distance percentiles; delta values are dataset-specific multiples of the base radius.",
        "- Fixed non-requested parameters: hamming_probe=2, min_collisions=2, annulus_count=1, ring_sample_size=16.",
        "- Feasible means point recall >= 0.80 and F1 >= 0.60.",
        "",
        "## Final Combined Parameters by Dataset",
        "",
        markdown_table(final_results, final_cols),
        "",
        "## Best Value Per Dataset and Requested Factor",
        "",
        markdown_table(best_results, best_cols),
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved multi-dataset sensitivity results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
