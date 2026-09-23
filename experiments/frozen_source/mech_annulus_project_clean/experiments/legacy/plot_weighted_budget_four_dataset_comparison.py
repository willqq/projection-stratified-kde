from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_DIR = Path("mech_annulus_count_weighted_budget_sensitivity")
CSV_PATH = IN_DIR / "all_annulus_count_weighted_budget.csv"
OUT_DIR = IN_DIR / "figures"
SUMMARY_PATH = IN_DIR / "four_dataset_weighted_budget_comparison_summary.md"

DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
ANNULUS_COUNTS = [1, 2, 4, 8, 16]
TOTAL_SAMPLE_BUDGETS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
BASELINE_BUDGET = 512
ERROR_TOLERANCE = 1.10
COLORS = {1: "#4c78a8", 2: "#f58518", 4: "#54a24b", 8: "#b279a2", 16: "#e45756"}


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
            elif column in {"annulus_count", "total_sample_budget", "best_annulus_count"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_best_by_budget(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        part = df[df["dataset"] == dataset]
        for budget in TOTAL_SAMPLE_BUDGETS:
            group = part[part["total_sample_budget"] == budget]
            best = group.sort_values(["kde_abs_relative_error", "query_time_ms"]).iloc[0]
            ring1 = group[group["annulus_count"] == 1].iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "total_sample_budget": budget,
                    "best_annulus_count": int(best["annulus_count"]),
                    "best_kde_error": float(best["kde_abs_relative_error"]),
                    "best_query_time_ms": float(best["query_time_ms"]),
                    "ring1_kde_error": float(ring1["kde_abs_relative_error"]),
                    "ring1_query_time_ms": float(ring1["query_time_ms"]),
                    "error_gain_vs_ring1": float(ring1["kde_abs_relative_error"] - best["kde_abs_relative_error"]),
                    "time_delta_vs_ring1": float(best["query_time_ms"] - ring1["query_time_ms"]),
                }
            )
    return pd.DataFrame(rows)


def build_close_faster_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        part = df[df["dataset"] == dataset]
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == BASELINE_BUDGET)
        ].iloc[0]
        threshold = float(baseline["kde_abs_relative_error"]) * ERROR_TOLERANCE
        candidates = part[
            (part["annulus_count"] > 1)
            & (part["kde_abs_relative_error"] <= threshold)
            & (part["query_time_ms"] < baseline["query_time_ms"])
        ].copy()
        if candidates.empty:
            best_larger = part[part["annulus_count"] > 1].sort_values(
                ["kde_abs_relative_error", "query_time_ms"]
            ).iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "selected_type": "best_larger_annulus_not_faster",
                    "annulus_count": int(best_larger["annulus_count"]),
                    "total_sample_budget": int(best_larger["total_sample_budget"]),
                    "kde_abs_relative_error": float(best_larger["kde_abs_relative_error"]),
                    "baseline_kde_error": float(baseline["kde_abs_relative_error"]),
                    "query_time_ms": float(best_larger["query_time_ms"]),
                    "baseline_query_time_ms": float(baseline["query_time_ms"]),
                    "kde_time_ms": float(best_larger["kde_time_ms"]),
                }
            )
        else:
            selected = candidates.sort_values(["query_time_ms", "kde_abs_relative_error"]).iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "selected_type": "close_density_faster",
                    "annulus_count": int(selected["annulus_count"]),
                    "total_sample_budget": int(selected["total_sample_budget"]),
                    "kde_abs_relative_error": float(selected["kde_abs_relative_error"]),
                    "baseline_kde_error": float(baseline["kde_abs_relative_error"]),
                    "query_time_ms": float(selected["query_time_ms"]),
                    "baseline_query_time_ms": float(baseline["query_time_ms"]),
                    "kde_time_ms": float(selected["kde_time_ms"]),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_lines(df: pd.DataFrame, metric: str, ylabel: str, out_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        part = df[df["dataset"] == dataset]
        for annulus_count in ANNULUS_COUNTS:
            group = part[part["annulus_count"] == annulus_count].sort_values("total_sample_budget")
            ax.plot(
                group["total_sample_budget"],
                group[metric],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=COLORS[annulus_count],
                label=str(annulus_count),
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(TOTAL_SAMPLE_BUDGETS)
        ax.set_xticklabels(TOTAL_SAMPLE_BUDGETS)
        ax.set_title(dataset)
        ax.set_xlabel("total sample budget")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="annulus count",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(ylabel + " under strict weighted total budget", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
    plt.close(fig)


def plot_pareto(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        part = df[df["dataset"] == dataset].copy()
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == BASELINE_BUDGET)
        ].iloc[0]
        threshold = baseline["kde_abs_relative_error"] * ERROR_TOLERANCE
        part["close_faster"] = (
            (part["annulus_count"] > 1)
            & (part["kde_abs_relative_error"] <= threshold)
            & (part["query_time_ms"] < baseline["query_time_ms"])
        )
        for annulus_count, group in part.groupby("annulus_count"):
            sizes = 34 + 1.2 * np.sqrt(group["total_sample_budget"].to_numpy(dtype=float))
            ax.scatter(
                group["query_time_ms"],
                group["kde_abs_relative_error"],
                s=sizes,
                alpha=0.82,
                color=COLORS[int(annulus_count)],
                edgecolor="white",
                linewidth=0.5,
                label=str(int(annulus_count)),
            )
        winners = part[part["close_faster"]]
        if not winners.empty:
            ax.scatter(
                winners["query_time_ms"],
                winners["kde_abs_relative_error"],
                s=170,
                facecolor="none",
                edgecolor="black",
                linewidth=1.4,
            )
        ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1.0)
        ax.axvline(baseline["query_time_ms"], color="#333333", linestyle=":", linewidth=1.0)
        ax.scatter(
            [baseline["query_time_ms"]],
            [baseline["kde_abs_relative_error"]],
            marker="*",
            s=180,
            color="black",
            label="baseline",
            zorder=5,
        )
        ax.set_title(dataset)
        ax.set_xlabel("query time ms")
        ax.set_ylabel("KDE relative error")
        ax.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle
    fig.legend(
        unique.values(),
        unique.keys(),
        title="annulus count",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(
        "Four-dataset KDE/time tradeoff under optimized weighted-budget annuli",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.savefig(OUT_DIR / "four_dataset_weighted_budget_pareto_tradeoff.png", dpi=260)
    fig.savefig(OUT_DIR / "four_dataset_weighted_budget_pareto_tradeoff.pdf")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    best_by_budget = build_best_by_budget(df)
    selected = build_close_faster_candidates(df)
    best_by_budget.to_csv(IN_DIR / "four_dataset_best_annulus_by_budget.csv", index=False)
    selected.to_csv(IN_DIR / "four_dataset_selected_tradeoff_points.csv", index=False)

    plot_metric_lines(
        df,
        "kde_abs_relative_error",
        "KDE relative error",
        "four_dataset_weighted_budget_kde_error_lines",
    )
    plot_metric_lines(
        df,
        "query_time_ms",
        "Online query time (ms)",
        "four_dataset_weighted_budget_query_time_lines",
    )
    plot_metric_lines(
        df,
        "kde_time_ms",
        "KDE-stage time (ms)",
        "four_dataset_weighted_budget_kde_time_lines",
    )
    plot_pareto(df)

    summary = [
        "# Four-Dataset Strict Weighted Total-Budget Comparison",
        "",
        "Setting: optimized radius-first MECH query, strict total KDE sample budget, and size-weighted allocation across annuli.",
        f"Baseline for close-density faster check: `annulus_count=1,total_sample_budget={BASELINE_BUDGET}`.",
        f"Close-density threshold: KDE relative error <= {ERROR_TOLERANCE:.2f}x baseline error.",
        "",
        "## Selected Tradeoff Points",
        "",
        markdown_table(
            selected.sort_values("dataset"),
            [
                "dataset",
                "selected_type",
                "annulus_count",
                "total_sample_budget",
                "kde_abs_relative_error",
                "baseline_kde_error",
                "query_time_ms",
                "baseline_query_time_ms",
                "kde_time_ms",
            ],
        ),
        "",
        "## Best Annulus Count By Budget",
        "",
        markdown_table(
            best_by_budget.sort_values(["dataset", "total_sample_budget"]),
            [
                "dataset",
                "total_sample_budget",
                "best_annulus_count",
                "best_kde_error",
                "best_query_time_ms",
                "ring1_kde_error",
                "ring1_query_time_ms",
                "error_gain_vs_ring1",
                "time_delta_vs_ring1",
            ],
        ),
        "",
        "## Figures",
        "",
        "- `figures/four_dataset_weighted_budget_kde_error_lines.png`",
        "- `figures/four_dataset_weighted_budget_query_time_lines.png`",
        "- `figures/four_dataset_weighted_budget_kde_time_lines.png`",
        "- `figures/four_dataset_weighted_budget_pareto_tradeoff.png`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
