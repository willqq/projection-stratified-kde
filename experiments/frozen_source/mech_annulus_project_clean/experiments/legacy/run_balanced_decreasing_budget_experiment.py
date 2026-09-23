from __future__ import annotations

"""Run the balanced decreasing-budget annulus experiment."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech_annulus_experiments as exp
import run_annulus_count_total_sample_sensitivity as base
import run_annulus_count_weighted_budget_sensitivity as weighted


OUT_DIR = Path("mech_annulus_balanced_decreasing_budget_2x_anchor")
OLD_SCHEDULE_PATH = (
    Path("mech_annulus_count_even_2_20_weighted_budget_sensitivity_2x_anchor_budget_10_500")
    / "decreasing_budget_schedules_10_500.csv"
)
DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
ANNULUS_COUNTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
SCHEDULE_BASE = 1000
BASELINE_ANNULUS = 2
BASELINE_BUDGET = 500
ALLOCATION_MODE = "balanced_size_weighted_total_budget"
ESTIMATOR_LABEL = "balanced size-weighted KDE"
FIGURE_PREFIX = "balanced"


def round_budget(value: float) -> int:
    return int(np.clip(round(value / 10.0) * 10, 10, 500))


def scheduled_budget(annulus_count: int) -> int:
    return round_budget(SCHEDULE_BASE / annulus_count)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, str) or isinstance(value, bool):
                cells.append(str(value))
            elif column in {"annulus_count", "total_sample_budget"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def add_baseline_ratios(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        baseline = part[
            (part["annulus_count"] == BASELINE_ANNULUS)
            & (part["total_sample_budget"] == BASELINE_BUDGET)
        ].iloc[0]
        current = part.copy()
        current["baseline_kde_error"] = float(baseline["kde_abs_relative_error"])
        current["baseline_query_time_ms"] = float(baseline["query_time_ms"])
        current["error_ratio_vs_baseline"] = (
            current["kde_abs_relative_error"] / current["baseline_kde_error"]
        )
        current["time_ratio_vs_baseline"] = (
            current["query_time_ms"] / current["baseline_query_time_ms"]
        )
        current["sample_ratio_vs_baseline"] = (
            current["total_sample_budget"] / BASELINE_BUDGET
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, dict]:
    exp.set_seed(exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    x = exp.load_dataset(dataset)
    sizes = base.split_sizes(len(x))
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
    base_radius = base.median_query_distance_percentile(distances, 35.0)
    query_radius = base.median_query_distance_percentile(
        distances,
        float(weighted.COMMON["radius_percentile"]),
    )
    norm_quantile = float(
        np.percentile(np.linalg.norm(index_x, axis=1), weighted.NORM_PERCENTILE)
    )
    delta = max(norm_quantile / weighted.DELTA_DIVISIONS, exp.EPS)

    model = exp.train_mech(
        train_x,
        weighted.MAX_TABLES,
        weighted.MAX_BITS,
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
        int(weighted.COMMON["L"]),
        int(weighted.COMMON["K"]),
        int(weighted.COMMON["hamming_probe"]),
    )

    rows = []
    for done, annulus_count in enumerate(ANNULUS_COUNTS, start=1):
        total_budget = scheduled_budget(annulus_count)
        config = exp.EvalConfig(
            delta=delta,
            radius_factor=1.0,
            bandwidth_factor=float(weighted.COMMON["bandwidth_factor"]),
            variant="full",
            query_mode="radius_first",
            min_collisions=int(weighted.COMMON["min_collisions"]),
            fixed_query_radius=query_radius,
            annulus_count=int(annulus_count),
            ring_sample_size=max(1, int(total_budget)),
            total_ring_sample_budget=int(total_budget),
            ring_sample_allocation=ALLOCATION_MODE,
        )
        start = time.perf_counter()
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
                "allocation_mode": ALLOCATION_MODE,
                "schedule_base": SCHEDULE_BASE,
                "strict_total_sample_budget": True,
                "norm_percentile": weighted.NORM_PERCENTILE,
                "delta_divisions": weighted.DELTA_DIVISIONS,
                "norm_quantile": norm_quantile,
                "delta": delta,
                "delta_factor_equiv": delta / base_radius,
                "base_radius": base_radius,
                "fixed_query_radius": query_radius,
                "L": weighted.COMMON["L"],
                "K": weighted.COMMON["K"],
                "radius_percentile": weighted.COMMON["radius_percentile"],
                "bandwidth_factor": weighted.COMMON["bandwidth_factor"],
                "hamming_probe": weighted.COMMON["hamming_probe"],
                "min_collisions": weighted.COMMON["min_collisions"],
                "wall_time_s": time.perf_counter() - start,
            }
        )
        rows.append(row)
        print(
            f"[{dataset} {done:02d}/{len(ANNULUS_COUNTS)}] "
            f"rings={annulus_count} budget={total_budget}: "
            f"KDE={row['kde_abs_relative_error']:.3f} "
            f"sample={row['kde_sample_size']:.1f} "
            f"T={row['query_time_ms']:.3f} "
            f"KDE_T={row['kde_time_ms']:.3f}",
            flush=True,
        )

    meta = {
        "dataset": dataset,
        "split": sizes,
        "base_radius": base_radius,
        "query_radius": query_radius,
        "norm_percentile": weighted.NORM_PERCENTILE,
        "delta_divisions": weighted.DELTA_DIVISIONS,
        "norm_quantile": norm_quantile,
        "delta": delta,
        "delta_factor_equiv": delta / base_radius,
        "bandwidth_base": bandwidth_base,
        "allocation_mode": ALLOCATION_MODE,
        "schedule_base": SCHEDULE_BASE,
    }
    return base.score_rows(pd.DataFrame(rows)), meta


def plot_normalized_effect(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    y_limit = max(
        2.6,
        float(df[["error_ratio_vs_baseline", "time_ratio_vs_baseline"]].max().max()) * 1.08,
    )
    colors = {
        "error_ratio_vs_baseline": "#d95f02",
        "time_ratio_vs_baseline": "#1b9e77",
        "sample_ratio_vs_baseline": "#4c78a8",
    }
    labels = {
        "error_ratio_vs_baseline": "KDE error / baseline",
        "time_ratio_vs_baseline": "query time / baseline",
        "sample_ratio_vs_baseline": "sample budget / baseline",
    }
    for ax, dataset in zip(axes, DATASETS):
        part = df[df["dataset"] == dataset].sort_values("annulus_count")
        for metric, color in colors.items():
            ax.plot(
                part["annulus_count"],
                part[metric],
                marker="o",
                linewidth=1.9,
                markersize=5.0,
                color=color,
                label=labels[metric],
            )
        for _, row in part.iterrows():
            ax.annotate(
                str(int(row["total_sample_budget"])),
                (row["annulus_count"], row["sample_ratio_vs_baseline"]),
                textcoords="offset points",
                xytext=(0, -13),
                ha="center",
                fontsize=7,
                color=colors["sample_ratio_vs_baseline"],
            )
        ax.axhline(1.0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.55)
        ax.set_title(dataset)
        ax.set_xlabel("annulus_count")
        ax.set_ylabel("ratio vs L=2, budget=500")
        ax.set_xticks(ANNULUS_COUNTS)
        ax.set_ylim(0.0, y_limit)
        ax.grid(True, alpha=0.22, linewidth=0.6)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center right", bbox_to_anchor=(0.99, 0.5))
    fig.suptitle(
        f"{ESTIMATOR_LABEL} with decreasing total sample budget",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.925,
        "Schedule: total_sample_budget = round_to_10(1000 / annulus_count). "
        "Blue labels show the actual total sample budget.",
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 0.80, 0.90))
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{FIGURE_PREFIX}_decreasing_budget_normalized.png", dpi=260)
    fig.savefig(figure_dir / f"{FIGURE_PREFIX}_decreasing_budget_normalized.pdf")
    plt.close(fig)


def plot_old_new_comparison(new_df: pd.DataFrame, old_df: pd.DataFrame) -> None:
    merged = new_df.merge(
        old_df[
            [
                "dataset",
                "annulus_count",
                "total_sample_budget",
                "kde_abs_relative_error",
                "query_time_ms",
                "kde_time_ms",
            ]
        ],
        on=["dataset", "annulus_count", "total_sample_budget"],
        suffixes=("_balanced", "_size_weighted"),
    )
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, out_name in [
        ("kde_abs_relative_error", "KDE relative error", f"old_vs_{FIGURE_PREFIX}_kde_error"),
        ("query_time_ms", "Online query time (ms)", f"old_vs_{FIGURE_PREFIX}_query_time"),
        ("kde_time_ms", "KDE-stage time (ms)", f"old_vs_{FIGURE_PREFIX}_kde_time"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
        axes = axes.ravel()
        for ax, dataset in zip(axes, DATASETS):
            part = merged[merged["dataset"] == dataset].sort_values("annulus_count")
            ax.plot(
                part["annulus_count"],
                part[f"{metric}_size_weighted"],
                marker="o",
                linewidth=1.8,
                color="#7570b3",
                label="size weighted",
            )
            ax.plot(
                part["annulus_count"],
                part[f"{metric}_balanced"],
                marker="o",
                linewidth=1.8,
                color="#d95f02",
                label="balanced",
            )
            ax.set_title(dataset)
            ax.set_xlabel("annulus_count")
            ax.set_ylabel(ylabel)
            ax.set_xticks(ANNULUS_COUNTS)
            ax.grid(True, alpha=0.22, linewidth=0.6)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.99, 0.5))
        fig.suptitle(
            ylabel + ": old size-weighted sampling vs balanced",
            x=0.02,
            ha="left",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 0.86, 0.94))
        fig.savefig(figure_dir / f"{out_name}.png", dpi=260)
        fig.savefig(figure_dir / f"{out_name}.pdf")
        plt.close(fig)
    merged.to_csv(OUT_DIR / "old_vs_balanced_decreasing_budget.csv", index=False)


def write_summary(df: pd.DataFrame, old_df: pd.DataFrame | None, metas: list[dict]) -> None:
    large = df[df["annulus_count"] >= 10].copy()
    selected = (
        large.sort_values(
            ["dataset", "error_ratio_vs_baseline", "time_ratio_vs_baseline"],
            ascending=[True, True, True],
        )
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    cols = [
        "dataset",
        "annulus_count",
        "total_sample_budget",
        "kde_abs_relative_error",
        "baseline_kde_error",
        "query_time_ms",
        "baseline_query_time_ms",
        "kde_sample_size",
        "error_ratio_vs_baseline",
        "time_ratio_vs_baseline",
        "sample_ratio_vs_baseline",
    ]
    lines = [
        "# Balanced Decreasing-Budget Annulus Experiment",
        "",
        "Estimator: every nonempty annulus receives a small minimum sample count, then the remaining budget is allocated by annulus size.",
        "`total_sample_budget = round_to_10(1000 / annulus_count)`.",
        "Baseline for ratios: `annulus_count=2,total_sample_budget=500` under the same estimator.",
        "",
        "## Best Large-Annulus Points",
        "",
        markdown_table(selected.sort_values("dataset"), cols),
        "",
        "## Full Decreasing-Budget Path",
        "",
        markdown_table(df.sort_values(["dataset", "annulus_count"]), cols),
        "",
        "## Figures",
        "",
        f"- `figures/{FIGURE_PREFIX}_decreasing_budget_normalized.png`",
        f"- `figures/old_vs_{FIGURE_PREFIX}_kde_error.png`",
        f"- `figures/old_vs_{FIGURE_PREFIX}_query_time.png`",
        f"- `figures/old_vs_{FIGURE_PREFIX}_kde_time.png`",
    ]
    if old_df is None:
        lines.append("")
        lines.append("Old size-weighted schedule results were not found, so old/new plots were skipped.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "annulus_counts": ANNULUS_COUNTS,
                "schedule_base": SCHEDULE_BASE,
                "allocation_mode": ALLOCATION_MODE,
                "common_parameters": weighted.COMMON,
                "norm_percentile": weighted.NORM_PERCENTILE,
                "delta_divisions": weighted.DELTA_DIVISIONS,
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metas = []
    for dataset in DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        df, meta = evaluate_dataset(dataset)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / "balanced_decreasing_budget.csv", index=False)
        all_rows.append(df)
        metas.append(meta)

    all_df = add_baseline_ratios(pd.concat(all_rows, ignore_index=True))
    all_df.to_csv(OUT_DIR / "balanced_decreasing_budget.csv", index=False)
    plot_normalized_effect(all_df)

    old_df = None
    if OLD_SCHEDULE_PATH.exists():
        old_df = pd.read_csv(OLD_SCHEDULE_PATH)
        old_df = old_df[old_df["schedule_base"] == SCHEDULE_BASE].copy()
        plot_old_new_comparison(all_df, old_df)
    write_summary(all_df, old_df, metas)
    print(f"Saved balanced decreasing-budget results to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
