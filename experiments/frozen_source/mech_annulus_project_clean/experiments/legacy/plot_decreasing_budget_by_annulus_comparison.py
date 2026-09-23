from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


IN_DIR = Path("mech_annulus_count_weighted_budget_sensitivity")
CSV_PATH = IN_DIR / "all_annulus_count_weighted_budget.csv"
OUT_DIR = IN_DIR / "figures"
SUMMARY_PATH = IN_DIR / "decreasing_budget_by_annulus_summary.md"

DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
ANNULUS_COUNTS = [1, 2, 4, 8, 16]
BASE_BUDGETS = [64, 128, 256, 512]
BASELINE_BUDGET = 512
ERROR_TOLERANCE = 1.10
COLORS = {64: "#4c78a8", 128: "#f58518", 256: "#54a24b", 512: "#e45756"}


def scheduled_budget(base_budget: int, annulus_count: int) -> int:
    return max(2, int(base_budget // annulus_count))


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
            elif column in {"annulus_count", "base_budget", "total_sample_budget"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_schedule_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        part = df[df["dataset"] == dataset]
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == BASELINE_BUDGET)
        ].iloc[0]
        for base_budget in BASE_BUDGETS:
            for annulus_count in ANNULUS_COUNTS:
                budget = scheduled_budget(base_budget, annulus_count)
                selected = part[
                    (part["annulus_count"] == annulus_count)
                    & (part["total_sample_budget"] == budget)
                ].iloc[0]
                rows.append(
                    {
                        "dataset": dataset,
                        "base_budget": base_budget,
                        "annulus_count": annulus_count,
                        "total_sample_budget": budget,
                        "kde_abs_relative_error": float(selected["kde_abs_relative_error"]),
                        "query_time_ms": float(selected["query_time_ms"]),
                        "kde_time_ms": float(selected["kde_time_ms"]),
                        "baseline_kde_error": float(baseline["kde_abs_relative_error"]),
                        "baseline_query_time_ms": float(baseline["query_time_ms"]),
                        "close_density": float(selected["kde_abs_relative_error"])
                        <= float(baseline["kde_abs_relative_error"]) * ERROR_TOLERANCE,
                        "faster_than_baseline": float(selected["query_time_ms"])
                        < float(baseline["query_time_ms"]),
                    }
                )
    out = pd.DataFrame(rows)
    out["close_and_faster"] = out["close_density"] & out["faster_than_baseline"]
    return out


def build_best_points(schedule_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        part = schedule_df[
            (schedule_df["dataset"] == dataset)
            & (schedule_df["annulus_count"] > 1)
        ].copy()
        good = part[part["close_and_faster"]]
        if not good.empty:
            selected = good.sort_values(["query_time_ms", "kde_abs_relative_error"]).iloc[0]
            selected_type = "decreasing_budget_close_faster"
        else:
            selected = part.sort_values(["kde_abs_relative_error", "query_time_ms"]).iloc[0]
            selected_type = "best_decreasing_budget_not_faster"
        current = selected.to_dict()
        current["selected_type"] = selected_type
        rows.append(current)
    return pd.DataFrame(rows)


def plot_schedule_metric(schedule_df: pd.DataFrame, metric: str, ylabel: str, out_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        part = schedule_df[schedule_df["dataset"] == dataset]
        for base_budget in BASE_BUDGETS:
            group = part[part["base_budget"] == base_budget].sort_values("annulus_count")
            ax.plot(
                group["annulus_count"],
                group[metric],
                marker="o",
                linewidth=1.8,
                markersize=4.8,
                color=COLORS[base_budget],
                label=f"B0={base_budget}",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ANNULUS_COUNTS)
        ax.set_xticklabels(ANNULUS_COUNTS)
        ax.set_title(dataset)
        ax.set_xlabel("annulus count")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="budget schedule", loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(ylabel + " when total samples decrease with annulus count", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.90, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
    plt.close(fig)


def plot_budget_schedule() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for base_budget in BASE_BUDGETS:
        budgets = [scheduled_budget(base_budget, annulus_count) for annulus_count in ANNULUS_COUNTS]
        ax.plot(
            ANNULUS_COUNTS,
            budgets,
            marker="o",
            linewidth=1.8,
            markersize=4.8,
            color=COLORS[base_budget],
            label=f"B0={base_budget}",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ANNULUS_COUNTS)
    ax.set_xticklabels(ANNULUS_COUNTS)
    ax.set_xlabel("annulus count")
    ax.set_ylabel("total sample budget")
    ax.set_title("Decreasing total sample budget schedule")
    ax.grid(alpha=0.18)
    ax.legend(title="budget schedule")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "decreasing_budget_schedule.png", dpi=260)
    fig.savefig(OUT_DIR / "decreasing_budget_schedule.pdf")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    schedule_df = build_schedule_frame(df)
    best_points = build_best_points(schedule_df)
    schedule_df.to_csv(IN_DIR / "decreasing_budget_by_annulus_paths.csv", index=False)
    best_points.to_csv(IN_DIR / "decreasing_budget_by_annulus_selected_points.csv", index=False)

    plot_budget_schedule()
    plot_schedule_metric(
        schedule_df,
        "kde_abs_relative_error",
        "KDE relative error",
        "decreasing_budget_kde_error_lines",
    )
    plot_schedule_metric(
        schedule_df,
        "query_time_ms",
        "Online query time (ms)",
        "decreasing_budget_query_time_lines",
    )
    plot_schedule_metric(
        schedule_df,
        "kde_time_ms",
        "KDE-stage time (ms)",
        "decreasing_budget_kde_time_lines",
    )

    selected_columns = [
        "dataset",
        "selected_type",
        "base_budget",
        "annulus_count",
        "total_sample_budget",
        "kde_abs_relative_error",
        "baseline_kde_error",
        "query_time_ms",
        "baseline_query_time_ms",
        "kde_time_ms",
    ]
    path_columns = [
        "dataset",
        "base_budget",
        "annulus_count",
        "total_sample_budget",
        "kde_abs_relative_error",
        "query_time_ms",
        "kde_time_ms",
        "close_and_faster",
    ]
    summary = [
        "# Decreasing Total Sample Budget By Annulus Count",
        "",
        "`total_sample_budget = max(2, B0 / annulus_count)`.",
        "This enforces fewer total KDE samples as `annulus_count` increases.",
        f"Baseline for close-density faster check: `annulus_count=1,total_sample_budget={BASELINE_BUDGET}`.",
        f"Close-density threshold: KDE relative error <= {ERROR_TOLERANCE:.2f}x baseline error.",
        "",
        "## Corrected Interpretation",
        "",
        "The decreasing-budget path does not support a general conclusion that larger `annulus_count` always preserves KDE accuracy while reducing query time.",
        "The evidence is dataset-dependent: Amazon supports close-density faster points, CIFAR10-GIST512 supports improved KDE error at some reduced budgets but not consistently lower query time, while CIFAR-10 and ISOLET show clear KDE-error degradation when the total sample budget is reduced with larger annulus counts.",
        "Therefore this experiment should be reported as a partial tradeoff result, not as a universal monotonic improvement.",
        "",
        "## Selected Points",
        "",
        markdown_table(best_points.sort_values("dataset"), selected_columns),
        "",
        "## Full Decreasing-Budget Paths",
        "",
        markdown_table(schedule_df.sort_values(["dataset", "base_budget", "annulus_count"]), path_columns),
        "",
        "## Figures",
        "",
        "- `figures/decreasing_budget_schedule.png`",
        "- `figures/decreasing_budget_kde_error_lines.png`",
        "- `figures/decreasing_budget_query_time_lines.png`",
        "- `figures/decreasing_budget_kde_time_lines.png`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
