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
SOURCE_DIR = Path("mech_multidataset_requested_sensitivity_dense")
OUT_DIR = Path("mech_common_optimal_multidataset_sensitivity")

SPLIT = {"n_train": 2800, "n_index": 1800, "n_queries": 40}
FIXED = {
    "hamming_probe": 2,
    "min_collisions": 2,
    "annulus_count": 1,
    "ring_sample_size": 16,
}

L_VALUES = list(range(2, 13))
K_VALUES = list(range(2, 13))
DELTA_FACTORS = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50]
RADIUS_PERCENTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]
BANDWIDTH_FACTORS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00]

FACTOR_SPECS = {
    "L": {"label": "Hash tables L", "values": L_VALUES, "x_label": "L"},
    "K": {"label": "Hash code length K", "values": K_VALUES, "x_label": "K"},
    "delta_factor": {
        "label": "Sphere interval factor",
        "values": DELTA_FACTORS,
        "x_label": "delta / base radius",
    },
    "radius_percentile": {
        "label": "Radius percentile",
        "values": RADIUS_PERCENTILES,
        "x_label": "distance percentile for R",
    },
    "bandwidth_factor": {
        "label": "Kernel bandwidth factor",
        "values": BANDWIDTH_FACTORS,
        "x_label": "kernel bandwidth factor",
    },
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
    return {"n_train": int(n_train), "n_index": int(n_index), "n_queries": int(n_queries)}


def radius_values(distances: np.ndarray) -> tuple[dict[int, float], float]:
    values = {}
    for percentile in RADIUS_PERCENTILES:
        per_query = np.percentile(distances, percentile, axis=1)
        values[percentile] = float(np.median(per_query))
    return values, values[35]


def add_common_axis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["common_factor"] = df["factor"].astype(str)
    df["common_value"] = df["factor_value"].astype(float)

    for dataset in df["dataset"].unique():
        mask_dataset = df["dataset"] == dataset

        delta_mask = mask_dataset & (df["factor"] == "delta")
        if delta_mask.any():
            values = sorted(df.loc[delta_mask, "factor_value"].astype(float).unique())
            mapping = {value: DELTA_FACTORS[i] for i, value in enumerate(values)}
            df.loc[delta_mask, "common_factor"] = "delta_factor"
            df.loc[delta_mask, "common_value"] = df.loc[delta_mask, "factor_value"].astype(float).map(mapping)

        radius_mask = mask_dataset & (df["factor"] == "fixed_query_radius")
        if radius_mask.any():
            df.loc[radius_mask, "common_factor"] = "radius_percentile"
            if "radius_percentile" in df.columns:
                df.loc[radius_mask, "common_value"] = df.loc[radius_mask, "radius_percentile"].astype(float)

    return df


def pick_common_parameters(source_dir: Path) -> tuple[dict[str, float], pd.DataFrame]:
    source = pd.read_csv(source_dir / "all_requested_parameter_sensitivity_dense.csv")
    source = add_common_axis(source)

    grouped = (
        source.groupby(["common_factor", "common_value"], as_index=False)
        .agg(
            mean_score_highR=("score_highR", "mean"),
            mean_score_balanced=("score_balanced", "mean"),
            mean_precision=("point_precision", "mean"),
            mean_recall=("point_recall", "mean"),
            mean_f1=("point_f1", "mean"),
            mean_kde_error=("kde_abs_relative_error", "mean"),
            mean_online_ms=("query_time_ms", "mean"),
            mean_cer=("candidate_expansion_ratio", "mean"),
            feasible_count=("feasible", "sum"),
        )
    )
    grouped["common_utility"] = grouped["mean_score_balanced"] - 0.03 * np.log1p(
        grouped["mean_cer"]
    )

    common = {}
    for factor in FACTOR_SPECS:
        part = grouped[grouped["common_factor"] == factor].copy()
        feasible = part[part["feasible_count"] >= 2]
        ranking = feasible if not feasible.empty else part
        best = ranking.sort_values(
            ["common_utility", "mean_f1", "mean_online_ms"],
            ascending=[False, False, True],
        ).iloc[0]
        common[factor] = float(best["common_value"])

    common["L"] = int(common["L"])
    common["K"] = int(common["K"])
    common["radius_percentile"] = int(common["radius_percentile"])
    return common, grouped


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


def evaluate_dataset(dataset: str, common: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
    radii, base_radius = radius_values(distances)

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

    index_cache = {}

    def get_index(tables: int, bits: int):
        key = (int(tables), int(bits), FIXED["hamming_probe"])
        if key not in index_cache:
            index_cache[key] = exp.MECHHashIndex(model, index_x, *key)
        return index_cache[key]

    rows = []
    total = sum(len(spec["values"]) for spec in FACTOR_SPECS.values())
    done = 0
    for factor, spec in FACTOR_SPECS.items():
        for value in spec["values"]:
            params = dict(common)
            params[factor] = value
            tables = int(params["L"])
            bits = int(params["K"])
            delta = float(params["delta_factor"]) * base_radius
            query_radius = radii[int(params["radius_percentile"])]
            bandwidth_factor = float(params["bandwidth_factor"])
            index = get_index(tables, bits)
            config = exp.EvalConfig(
                delta=delta,
                radius_factor=1.0,
                bandwidth_factor=bandwidth_factor,
                variant="full",
                query_mode="radius_first",
                min_collisions=FIXED["min_collisions"],
                fixed_query_radius=query_radius,
                annulus_count=FIXED["annulus_count"],
                ring_sample_size=FIXED["ring_sample_size"],
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
            row.update(
                {
                    "dataset": dataset,
                    "factor": factor,
                    "factor_label": spec["label"],
                    "common_value": value,
                    "L": tables,
                    "K": bits,
                    "delta_factor": float(params["delta_factor"]),
                    "radius_percentile": int(params["radius_percentile"]),
                    "fixed_query_radius": query_radius,
                    "bandwidth_factor": bandwidth_factor,
                    "hamming_probe": FIXED["hamming_probe"],
                    "min_collisions": FIXED["min_collisions"],
                    "annulus_count": FIXED["annulus_count"],
                    "ring_sample_size": FIXED["ring_sample_size"],
                }
            )
            rows.append(row)
            done += 1
            print(
                f"[{dataset} {done:02d}/{total}] {factor}={value}: "
                f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                f"F1={row['point_f1']:.3f} KDE={row['kde_abs_relative_error']:.3f} "
                f"T={row['query_time_ms']:.3f}",
                flush=True,
            )

    df = score_rows(pd.DataFrame(rows))
    baseline = df[
        (df["factor"] == "L")
        & (df["common_value"].astype(float) == float(common["L"]))
    ].copy()
    meta = {
        "dataset": dataset,
        "split": sizes,
        "base_radius": base_radius,
        "radii": radii,
        "bandwidth_base": bandwidth_base,
    }
    return df, baseline, meta


def plot_common_factor_curves(df: pd.DataFrame, out_dir: Path) -> None:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("point_recall", "Recall"),
        ("point_f1", "F1"),
        ("kde_abs_relative_error", "KDE Error"),
        ("query_time_ms", "Online ms"),
    ]
    colors = {
        "isolet": "#2F6B9A",
        "cifar10": "#2F8F83",
        "cifar10_gist512": "#D9A441",
        "amazon": "#B55252",
    }
    for factor, spec in FACTOR_SPECS.items():
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
        axes = axes.ravel()
        for ax, (metric, title) in zip(axes, metrics):
            for dataset in DATASETS:
                part = df[(df["dataset"] == dataset) & (df["factor"] == factor)].sort_values("common_value")
                ax.plot(
                    part["common_value"].astype(float),
                    part[metric].astype(float),
                    marker="o",
                    linewidth=1.8,
                    label=dataset,
                    color=colors[dataset],
                )
            ax.set_title(title)
            ax.set_xlabel(spec["x_label"])
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Common-baseline sensitivity: {spec['label']}", x=0.02, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{factor}_common_comparison.png", dpi=240)
        fig.savefig(figure_dir / f"{factor}_common_comparison.pdf")
        plt.close(fig)


def format_cell(column: str, value) -> str:
    if column in {"dataset", "factor", "factor_label", "feasible"}:
        return str(value)
    if column in {"L", "K", "hamming_probe", "min_collisions", "annulus_count", "ring_sample_size", "radius_percentile"}:
        return str(int(value))
    if column in {"delta_factor", "common_value", "fixed_query_radius", "bandwidth_factor", "delta"}:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common, aggregated = pick_common_parameters(SOURCE_DIR)
    aggregated.to_csv(OUT_DIR / "common_parameter_selection_scores.csv", index=False)

    print("Common optimal parameters:", common, flush=True)

    all_rows = []
    baselines = []
    metas = []
    for dataset in DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        df, baseline, meta = evaluate_dataset(dataset, common)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / "common_baseline_sensitivity.csv", index=False)
        baseline.to_csv(dataset_dir / "common_baseline_row.csv", index=False)
        all_rows.append(df)
        baselines.append(baseline)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    baseline_df = pd.concat(baselines, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_common_baseline_sensitivity.csv", index=False)
    baseline_df.to_csv(OUT_DIR / "common_baseline_by_dataset.csv", index=False)
    plot_common_factor_curves(all_df, OUT_DIR)

    baseline_cols = [
        "dataset",
        "L",
        "K",
        "delta_factor",
        "radius_percentile",
        "fixed_query_radius",
        "bandwidth_factor",
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
    summary = [
        "# Common-Optimal Multi-Dataset Sensitivity",
        "",
        "## Common Parameters",
        "",
        "```text",
        f"L = {common['L']}",
        f"K = {common['K']}",
        f"delta_factor = {common['delta_factor']}",
        f"radius_percentile = {common['radius_percentile']}",
        f"bandwidth_factor = {common['bandwidth_factor']}",
        "hamming_probe = 2",
        "min_collisions = 2",
        "annulus_count = 1",
        "ring_sample_size = 16",
        "```",
        "",
        "## Common-Baseline Result By Dataset",
        "",
        markdown_table(baseline_df, baseline_cols),
        "",
        "## Plots",
        "",
        "- `figures/L_common_comparison.png`",
        "- `figures/K_common_comparison.png`",
        "- `figures/delta_factor_common_comparison.png`",
        "- `figures/radius_percentile_common_comparison.png`",
        "- `figures/bandwidth_factor_common_comparison.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "common_parameters": common,
                "fixed_parameters": FIXED,
                "factor_specs": {k: v["values"] for k, v in FACTOR_SPECS.items()},
                "dataset_meta": metas,
                "source_dir": str(SOURCE_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved common-optimal sensitivity results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
