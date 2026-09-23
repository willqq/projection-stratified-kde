from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_DIR = Path(
    "mech_annulus_count_even_2_20_weighted_budget_sensitivity_2x_anchor_budget_10_500"
)
CSV_PATH = IN_DIR / "all_annulus_count_even_2_20_weighted_budget.csv"
OUT_DIR = IN_DIR / "figures"
SUMMARY_PATH = IN_DIR / "large_annulus_low_budget_effect_summary.md"

DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
ANNULUS_COUNTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
BASELINE_ANNULUS = 2
BASELINE_BUDGET = 500
LARGE_ANNULUS_MIN = 10
LOW_BUDGET_MAX = 100
ERROR_TOLERANCE = 1.25
SCHEDULE_BASES = [200, 500, 1000]


def round_budget(value: float) -> int:
    return int(np.clip(round(value / 10.0) * 10, 10, 500))


def scheduled_budget(base_budget: int, annulus_count: int) -> int:
    return round_budget(base_budget / annulus_count)


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
            elif column in {
                "annulus_count",
                "total_sample_budget",
                "baseline_annulus_count",
                "baseline_total_sample_budget",
                "schedule_base",
            }:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def add_baseline_columns(part: pd.DataFrame) -> pd.DataFrame:
    baseline = part[
        (part["annulus_count"] == BASELINE_ANNULUS)
        & (part["total_sample_budget"] == BASELINE_BUDGET)
    ].iloc[0]
    current = part.copy()
    current["baseline_annulus_count"] = BASELINE_ANNULUS
    current["baseline_total_sample_budget"] = BASELINE_BUDGET
    current["baseline_kde_error"] = float(baseline["kde_abs_relative_error"])
    current["baseline_query_time_ms"] = float(baseline["query_time_ms"])
    current["error_ratio_vs_baseline"] = (
        current["kde_abs_relative_error"] / current["baseline_kde_error"]
    )
    current["time_ratio_vs_baseline"] = (
        current["query_time_ms"] / current["baseline_query_time_ms"]
    )
    current["sample_ratio_vs_baseline"] = (
        current["total_sample_budget"] / current["baseline_total_sample_budget"]
    )
    current["close_density"] = current["error_ratio_vs_baseline"] <= ERROR_TOLERANCE
    current["faster_than_baseline"] = current["time_ratio_vs_baseline"] < 1.0
    return current


def build_large_low_budget_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset in DATASETS:
        part = add_baseline_columns(df[df["dataset"] == dataset])
        candidates = part[
            (part["annulus_count"] >= LARGE_ANNULUS_MIN)
            & (part["total_sample_budget"] <= LOW_BUDGET_MAX)
        ].copy()
        candidates["large_low_budget"] = True
        rows.append(candidates)
    all_candidates = pd.concat(rows, ignore_index=True)

    selected_rows = []
    for dataset, part in all_candidates.groupby("dataset"):
        close_faster = part[part["close_density"] & part["faster_than_baseline"]]
        if not close_faster.empty:
            selected = close_faster.sort_values(
                ["total_sample_budget", "kde_abs_relative_error", "query_time_ms"]
            ).iloc[0]
            selected_type = "large_low_budget_close_faster"
        else:
            selected = part.sort_values(
                ["kde_abs_relative_error", "query_time_ms", "total_sample_budget"]
            ).iloc[0]
            selected_type = "best_large_low_budget_not_close_faster"
        current = selected.to_dict()
        current["selected_type"] = selected_type
        selected_rows.append(current)
    selected = pd.DataFrame(selected_rows)
    return all_candidates, selected


def build_schedule_paths(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        part = add_baseline_columns(df[df["dataset"] == dataset])
        for schedule_base in SCHEDULE_BASES:
            for annulus_count in ANNULUS_COUNTS:
                budget = scheduled_budget(schedule_base, annulus_count)
                selected = part[
                    (part["annulus_count"] == annulus_count)
                    & (part["total_sample_budget"] == budget)
                ].iloc[0]
                row = selected.to_dict()
                row["schedule_base"] = schedule_base
                rows.append(row)
    return pd.DataFrame(rows)


def plot_large_low_budget_scatter(candidates: pd.DataFrame, selected: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    axes = axes.ravel()
    colors = {
        10: "#4c78a8",
        12: "#f58518",
        14: "#54a24b",
        16: "#b279a2",
        18: "#e45756",
        20: "#72b7b2",
    }
    for ax, dataset in zip(axes, DATASETS):
        part = candidates[candidates["dataset"] == dataset]
        for annulus_count, group in part.groupby("annulus_count"):
            ax.scatter(
                group["query_time_ms"],
                group["kde_abs_relative_error"],
                s=35 + group["total_sample_budget"] * 0.45,
                alpha=0.82,
                color=colors[int(annulus_count)],
                edgecolor="white",
                linewidth=0.5,
                label=str(int(annulus_count)),
            )
        chosen = selected[selected["dataset"] == dataset]
        ax.scatter(
            chosen["query_time_ms"],
            chosen["kde_abs_relative_error"],
            s=210,
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
        )
        baseline_error = float(part["baseline_kde_error"].iloc[0])
        baseline_time = float(part["baseline_query_time_ms"].iloc[0])
        ax.axhline(baseline_error * ERROR_TOLERANCE, color="#333333", linestyle="--", linewidth=1.0)
        ax.axvline(baseline_time, color="#333333", linestyle=":", linewidth=1.0)
        ax.set_title(dataset)
        ax.set_xlabel("query time ms")
        ax.set_ylabel("KDE relative error")
        ax.grid(True, alpha=0.22, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        title="annulus_count",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(
        "Large annulus count with low total sample budget",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "large_annulus_low_budget_tradeoff.png", dpi=260)
    fig.savefig(OUT_DIR / "large_annulus_low_budget_tradeoff.pdf")
    plt.close(fig)


def plot_schedule_paths(schedule_df: pd.DataFrame, metric: str, ylabel: str, out_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    axes = axes.ravel()
    colors = {200: "#4c78a8", 500: "#f58518", 1000: "#54a24b"}
    for ax, dataset in zip(axes, DATASETS):
        part = schedule_df[schedule_df["dataset"] == dataset]
        for schedule_base in SCHEDULE_BASES:
            group = part[part["schedule_base"] == schedule_base].sort_values("annulus_count")
            ax.plot(
                group["annulus_count"],
                group[metric],
                marker="o",
                linewidth=1.7,
                markersize=4.8,
                color=colors[schedule_base],
                label=f"B0={schedule_base}",
            )
            for _, row in group.iterrows():
                ax.annotate(
                    str(int(row["total_sample_budget"])),
                    (row["annulus_count"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=7,
                    color=colors[schedule_base],
                )
        ax.set_title(dataset)
        ax.set_xlabel("annulus_count")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ANNULUS_COUNTS)
        ax.grid(True, alpha=0.22, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="budget=B0/L", loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(
        ylabel + " under decreasing total-sample schedules",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.90, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
    plt.close(fig)


def plot_normalized_decreasing_effect(schedule_df: pd.DataFrame) -> None:
    schedule_base = 1000
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
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
        part = schedule_df[
            (schedule_df["dataset"] == dataset)
            & (schedule_df["schedule_base"] == schedule_base)
        ].sort_values("annulus_count")
        for metric in colors:
            ax.plot(
                part["annulus_count"],
                part[metric],
                marker="o",
                linewidth=1.9,
                markersize=5.0,
                color=colors[metric],
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
        ax.axhline(ERROR_TOLERANCE, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(dataset)
        ax.set_xlabel("annulus_count")
        ax.set_ylabel("ratio vs L=2, budget=500")
        ax.set_xticks(ANNULUS_COUNTS)
        ax.set_ylim(0.0, 3.05)
        ax.grid(True, alpha=0.22, linewidth=0.6)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="center right", bbox_to_anchor=(0.99, 0.5))
    fig.suptitle(
        "Larger annulus count with decreasing total sample budget",
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "decreasing_schedule_normalized_effect.png", dpi=260)
    fig.savefig(OUT_DIR / "decreasing_schedule_normalized_effect.pdf")
    plt.close(fig)


def write_summary(candidates: pd.DataFrame, selected: pd.DataFrame, schedule_df: pd.DataFrame) -> None:
    selected_cols = [
        "dataset",
        "selected_type",
        "annulus_count",
        "total_sample_budget",
        "kde_abs_relative_error",
        "baseline_kde_error",
        "query_time_ms",
        "baseline_query_time_ms",
        "error_ratio_vs_baseline",
        "time_ratio_vs_baseline",
        "sample_ratio_vs_baseline",
    ]
    schedule_cols = [
        "dataset",
        "schedule_base",
        "annulus_count",
        "total_sample_budget",
        "kde_abs_relative_error",
        "query_time_ms",
        "error_ratio_vs_baseline",
        "time_ratio_vs_baseline",
        "sample_ratio_vs_baseline",
    ]
    lines = [
        "# Large Annulus With Lower Sampling Budget",
        "",
        f"Baseline: `annulus_count={BASELINE_ANNULUS}, total_sample_budget={BASELINE_BUDGET}` for each dataset.",
        f"Large-low-budget candidate region: `annulus_count >= {LARGE_ANNULUS_MIN}` and `total_sample_budget <= {LOW_BUDGET_MAX}`.",
        f"Close density threshold: KDE error <= {ERROR_TOLERANCE:.2f}x baseline error.",
        "",
        "## Selected Large-Low-Budget Points",
        "",
        markdown_table(selected.sort_values("dataset"), selected_cols),
        "",
        "## Interpretation",
        "",
        "- Amazon has valid large-annulus, low-sample points that are both close-density and faster than the small-annulus high-budget baseline.",
        "- CIFAR-10, CIFAR10-GIST512, and ISOLET do not satisfy the close-and-faster condition in the strict large-low-budget region; reducing samples with larger annulus counts either increases KDE error, increases time, or both.",
        "- The effect therefore exists, but it is dataset-dependent. It should not be written as a universal monotonic claim.",
        "",
        "## Decreasing-Budget Schedules",
        "",
        "`total_sample_budget = round_to_10(B0 / annulus_count)`, with `B0 in {200,500,1000}`.",
        "Numbers annotated on the schedule figures are the actual total sample budgets.",
        "",
        markdown_table(
            schedule_df.sort_values(["dataset", "schedule_base", "annulus_count"]),
            schedule_cols,
        ),
        "",
        "## Figures",
        "",
        "- `figures/large_annulus_low_budget_tradeoff.png`",
        "- `figures/decreasing_schedule_kde_error.png`",
        "- `figures/decreasing_schedule_query_time.png`",
        "- `figures/decreasing_schedule_kde_time.png`",
        "- `figures/decreasing_schedule_normalized_effect.png`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    candidates.to_csv(IN_DIR / "large_annulus_low_budget_candidates.csv", index=False)
    selected.to_csv(IN_DIR / "large_annulus_low_budget_selected.csv", index=False)
    schedule_df.to_csv(IN_DIR / "decreasing_budget_schedules_10_500.csv", index=False)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    candidates, selected = build_large_low_budget_candidates(df)
    schedule_df = build_schedule_paths(df)
    plot_large_low_budget_scatter(candidates, selected)
    plot_schedule_paths(
        schedule_df,
        "kde_abs_relative_error",
        "KDE relative error",
        "decreasing_schedule_kde_error",
    )
    plot_schedule_paths(
        schedule_df,
        "query_time_ms",
        "Online query time (ms)",
        "decreasing_schedule_query_time",
    )
    plot_schedule_paths(
        schedule_df,
        "kde_time_ms",
        "KDE-stage time (ms)",
        "decreasing_schedule_kde_time",
    )
    plot_normalized_decreasing_effect(schedule_df)
    write_summary(candidates, selected, schedule_df)


if __name__ == "__main__":
    main()
