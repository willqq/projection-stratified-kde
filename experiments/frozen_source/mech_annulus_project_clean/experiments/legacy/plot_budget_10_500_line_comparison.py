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
BEST_PATH = IN_DIR / "best_even_2_20_annulus_by_budget.csv"
OUT_DIR = IN_DIR / "figures"


def plot_metric_lines(df: pd.DataFrame, metric: str, ylabel: str, out_name: str) -> None:
    datasets = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
    annulus_counts = sorted(df["annulus_count"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(annulus_counts)))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.8), sharex=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets):
        part = df[df["dataset"] == dataset]
        for color, annulus_count in zip(colors, annulus_counts):
            curve = part[part["annulus_count"] == annulus_count].sort_values(
                "total_sample_budget"
            )
            ax.plot(
                curve["total_sample_budget"],
                curve[metric],
                color=color,
                linewidth=1.4,
                alpha=0.9,
                label=str(int(annulus_count)),
            )
        ax.set_title(dataset)
        ax.set_xlabel("total sample budget")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="annulus_count",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(ylabel + " under budget 10-500 sensitivity", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
    plt.close(fig)


def plot_best_annulus(best_df: pd.DataFrame) -> None:
    datasets = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets):
        part = best_df[best_df["dataset"] == dataset].sort_values("total_sample_budget")
        ax.step(
            part["total_sample_budget"],
            part["best_annulus_count"],
            where="mid",
            linewidth=1.6,
            color="#1f77b4",
        )
        ax.scatter(
            part["total_sample_budget"],
            part["best_annulus_count"],
            s=12,
            color="#1f77b4",
            alpha=0.75,
        )
        ax.set_title(dataset)
        ax.set_xlabel("total sample budget")
        ax.set_ylabel("best annulus_count")
        ax.set_yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        ax.grid(True, alpha=0.25, linewidth=0.6)

    fig.suptitle("Best annulus_count by total sample budget", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "budget_10_500_best_annulus_by_budget_lines.png", dpi=260)
    fig.savefig(OUT_DIR / "budget_10_500_best_annulus_by_budget_lines.pdf")
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
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(df: pd.DataFrame, best_df: pd.DataFrame) -> None:
    best_grid = (
        df.sort_values(
            ["dataset", "score_kde", "kde_abs_relative_error", "query_time_ms"],
            ascending=[True, False, True, True],
        )
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    min_error = (
        df.sort_values(["dataset", "kde_abs_relative_error", "query_time_ms"])
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    lines = [
        "# Budget 10-500 Line Comparison",
        "",
        "Total sample budgets are `10,20,...,500`; annulus counts are `2,4,...,20`.",
        "",
        "## Best Composite Score By Dataset",
        "",
        markdown_table(
            best_grid[
                [
                    "dataset",
                    "annulus_count",
                    "total_sample_budget",
                    "kde_abs_relative_error",
                    "query_time_ms",
                    "score_kde",
                ]
            ],
            [
                "dataset",
                "annulus_count",
                "total_sample_budget",
                "kde_abs_relative_error",
                "query_time_ms",
                "score_kde",
            ],
        ),
        "",
        "## Lowest KDE Error By Dataset",
        "",
        markdown_table(
            min_error[
                [
                    "dataset",
                    "annulus_count",
                    "total_sample_budget",
                    "kde_abs_relative_error",
                    "query_time_ms",
                    "score_kde",
                ]
            ],
            [
                "dataset",
                "annulus_count",
                "total_sample_budget",
                "kde_abs_relative_error",
                "query_time_ms",
                "score_kde",
            ],
        ),
        "",
        "## Figures",
        "",
        "- `figures/budget_10_500_kde_error_lines.png`",
        "- `figures/budget_10_500_query_time_lines.png`",
        "- `figures/budget_10_500_kde_time_lines.png`",
        "- `figures/budget_10_500_best_annulus_by_budget_lines.png`",
    ]
    (IN_DIR / "budget_10_500_line_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    best_df = pd.read_csv(BEST_PATH)
    plot_metric_lines(
        df,
        "kde_abs_relative_error",
        "KDE relative error",
        "budget_10_500_kde_error_lines",
    )
    plot_metric_lines(
        df,
        "query_time_ms",
        "Online query time (ms)",
        "budget_10_500_query_time_lines",
    )
    plot_metric_lines(
        df,
        "kde_time_ms",
        "KDE-stage time (ms)",
        "budget_10_500_kde_time_lines",
    )
    plot_best_annulus(best_df)
    write_summary(df, best_df)


if __name__ == "__main__":
    main()
