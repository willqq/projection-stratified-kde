from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech_annulus_experiments as exp


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("mech_annulus_count_total_sample_sensitivity")

SPLIT = {"n_train": 2800, "n_index": 1800, "n_queries": 40}
MAX_TABLES = 18
MAX_BITS = 12
COMMON = {
    "L": 12,
    "K": 5,
    "radius_percentile": 70,
    "bandwidth_factor": 0.25,
    "hamming_probe": 2,
    "min_collisions": 2,
}

NORM_PERCENTILE = 25
DELTA_DIVISIONS = 12
ANNULUS_COUNTS = [1, 2, 4, 8, 16]
TOTAL_SAMPLE_BUDGETS = [2, 4, 8, 16, 32, 64, 128, 256, 512]


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


def median_query_distance_percentile(distances: np.ndarray, percentile: float) -> float:
    return float(np.median(np.percentile(distances, percentile, axis=1)))


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kde_quality"] = (1.0 - df["kde_abs_relative_error"]).clip(lower=0.0, upper=1.0)
    time_max = max(float(df["query_time_ms"].max()), 1e-12)
    df["speed_quality"] = (1.0 - df["query_time_ms"] / time_max).clip(lower=0.0, upper=1.0)
    df["score_kde"] = (
        0.55 * df["kde_quality"]
        + 0.20 * df["point_recall"]
        + 0.15 * df["point_f1"]
        + 0.10 * df["speed_quality"]
    )
    df["feasible"] = (df["point_recall"] >= 0.80) & (df["point_f1"] >= 0.60)
    return df


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, dict]:
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
    base_radius = median_query_distance_percentile(distances, 35.0)
    query_radius = median_query_distance_percentile(
        distances,
        float(COMMON["radius_percentile"]),
    )
    norm_quantile = float(np.percentile(np.linalg.norm(index_x, axis=1), NORM_PERCENTILE))
    delta = max(norm_quantile / DELTA_DIVISIONS, exp.EPS)

    model = exp.train_mech(
        train_x,
        MAX_TABLES,
        MAX_BITS,
        24,
        128,
        1e-3,
        1.0,
        0.1,
        0.1,
        0.1,
        "cpu",
    )
    index = exp.MECHHashIndex(
        model,
        index_x,
        int(COMMON["L"]),
        int(COMMON["K"]),
        int(COMMON["hamming_probe"]),
    )

    rows = []
    total = len(ANNULUS_COUNTS) * len(TOTAL_SAMPLE_BUDGETS)
    done = 0
    for annulus_count in ANNULUS_COUNTS:
        for total_budget in TOTAL_SAMPLE_BUDGETS:
            per_ring_sample = max(1, int(math.ceil(total_budget / annulus_count)))
            config = exp.EvalConfig(
                delta=delta,
                radius_factor=1.0,
                bandwidth_factor=float(COMMON["bandwidth_factor"]),
                variant="full",
                query_mode="radius_first",
                min_collisions=int(COMMON["min_collisions"]),
                fixed_query_radius=query_radius,
                annulus_count=int(annulus_count),
                ring_sample_size=per_ring_sample,
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
                    "annulus_count": int(annulus_count),
                    "total_sample_budget": int(total_budget),
                    "per_ring_sample_size": int(per_ring_sample),
                    "norm_percentile": NORM_PERCENTILE,
                    "delta_divisions": DELTA_DIVISIONS,
                    "norm_quantile": norm_quantile,
                    "delta": delta,
                    "delta_factor_equiv": delta / base_radius,
                    "base_radius": base_radius,
                    "fixed_query_radius": query_radius,
                    "L": COMMON["L"],
                    "K": COMMON["K"],
                    "radius_percentile": COMMON["radius_percentile"],
                    "bandwidth_factor": COMMON["bandwidth_factor"],
                    "hamming_probe": COMMON["hamming_probe"],
                    "min_collisions": COMMON["min_collisions"],
                }
            )
            rows.append(row)
            done += 1
            print(
                f"[{dataset} {done:02d}/{total}] "
                f"rings={annulus_count} total={total_budget} per_ring={per_ring_sample}: "
                f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                f"F1={row['point_f1']:.3f} KDE={row['kde_abs_relative_error']:.3f} "
                f"actual_sample={row['kde_sample_size']:.1f} T={row['query_time_ms']:.3f}",
                flush=True,
            )

    meta = {
        "dataset": dataset,
        "split": sizes,
        "base_radius": base_radius,
        "query_radius": query_radius,
        "norm_percentile": NORM_PERCENTILE,
        "delta_divisions": DELTA_DIVISIONS,
        "norm_quantile": norm_quantile,
        "delta": delta,
        "delta_factor_equiv": delta / base_radius,
        "bandwidth_base": bandwidth_base,
    }
    return score_rows(pd.DataFrame(rows)), meta


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.pivot_table(
            index="total_sample_budget",
            columns="annulus_count",
            values=metric,
            aggfunc="mean",
        )
        .reindex(index=TOTAL_SAMPLE_BUDGETS, columns=ANNULUS_COUNTS)
    )


def add_heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, cmap: str) -> None:
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(ANNULUS_COUNTS)))
    ax.set_xticklabels(ANNULUS_COUNTS)
    ax.set_yticks(np.arange(len(TOTAL_SAMPLE_BUDGETS)))
    ax.set_yticklabels(TOTAL_SAMPLE_BUDGETS)
    ax.set_xlabel("annulus count")
    ax.set_ylabel("total sample budget")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_dataset_heatmaps(df: pd.DataFrame, dataset: str) -> None:
    figure_dir = OUT_DIR / "figures" / dataset
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("kde_abs_relative_error", "KDE relative error", "viridis"),
        ("signed_kde_bias", "Signed KDE bias", "coolwarm"),
        ("query_time_ms", "Online query time (ms)", "magma"),
        ("kde_sample_size", "Actual KDE sample size", "cividis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    axes = axes.ravel()
    for ax, (metric, title, cmap) in zip(axes, metrics):
        add_heatmap(ax, pivot_metric(df, metric), title, cmap)
    fig.suptitle(
        f"{dataset}: annulus count and total sample-budget sensitivity",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "annulus_count_total_sample_heatmaps.png", dpi=260)
    fig.savefig(figure_dir / "annulus_count_total_sample_heatmaps.pdf")
    plt.close(fig)


def plot_cross_dataset_metric(all_df: pd.DataFrame, metric: str, title: str, out_name: str, cmap: str) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        part = all_df[all_df["dataset"] == dataset]
        add_heatmap(ax, pivot_metric(part, metric), dataset, cmap)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{out_name}.png", dpi=260)
    fig.savefig(figure_dir / f"{out_name}.pdf")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                cells.append(value)
            elif column in {"annulus_count", "total_sample_budget", "per_ring_sample_size", "L", "K"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metas = []
    for dataset in DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        df, meta = evaluate_dataset(dataset)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / "annulus_count_total_sample.csv", index=False)
        plot_dataset_heatmaps(df, dataset)
        all_rows.append(df)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_annulus_count_total_sample.csv", index=False)
    plot_cross_dataset_metric(
        all_df,
        "kde_abs_relative_error",
        "KDE relative error under annulus count and total sample-budget sensitivity",
        "kde_error_cross_dataset_heatmaps",
        "viridis",
    )
    plot_cross_dataset_metric(
        all_df,
        "query_time_ms",
        "Online time under annulus count and total sample-budget sensitivity",
        "online_time_cross_dataset_heatmaps",
        "magma",
    )

    best_rows = []
    baseline_rows = []
    for dataset, part in all_df.groupby("dataset"):
        feasible = part[part["feasible"]]
        pool = feasible if not feasible.empty else part
        best = pool.sort_values(
            ["score_kde", "kde_abs_relative_error", "query_time_ms"],
            ascending=[False, True, True],
        ).iloc[0]
        best_rows.append(best)
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == 16)
        ].iloc[0]
        baseline_rows.append(baseline)

    best_df = pd.DataFrame(best_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    best_df.to_csv(OUT_DIR / "best_annulus_count_total_sample_by_dataset.csv", index=False)
    baseline_df.to_csv(OUT_DIR / "baseline_annulus_count_total_sample_by_dataset.csv", index=False)

    cols = [
        "dataset",
        "annulus_count",
        "total_sample_budget",
        "per_ring_sample_size",
        "kde_sample_size",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "signed_kde_bias",
        "candidate_expansion_ratio",
        "query_time_ms",
        "score_kde",
    ]
    summary = [
        "# Annulus Count And Total Sample-Budget Sensitivity",
        "",
        "This experiment tests whether increasing the number of annuli allows fewer samples per annulus under the same total sample budget.",
        f"`delta` is fixed as `Q_{NORM_PERCENTILE}(||x||) / {DELTA_DIVISIONS}`.",
        "`per_ring_sample_size = ceil(total_sample_budget / annulus_count)`.",
        "",
        "## Baseline Grid Point",
        "",
        markdown_table(baseline_df.sort_values("dataset"), cols),
        "",
        "## Best Grid Point By Dataset",
        "",
        markdown_table(best_df.sort_values("dataset"), cols),
        "",
        "## Figures",
        "",
        "- `figures/kde_error_cross_dataset_heatmaps.png`",
        "- `figures/online_time_cross_dataset_heatmaps.png`",
        "- `figures/<dataset>/annulus_count_total_sample_heatmaps.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "common_parameters": COMMON,
                "max_tables": MAX_TABLES,
                "max_bits": MAX_BITS,
                "norm_percentile": NORM_PERCENTILE,
                "delta_divisions": DELTA_DIVISIONS,
                "annulus_counts": ANNULUS_COUNTS,
                "total_sample_budgets": TOTAL_SAMPLE_BUDGETS,
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved annulus-count sensitivity results to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
