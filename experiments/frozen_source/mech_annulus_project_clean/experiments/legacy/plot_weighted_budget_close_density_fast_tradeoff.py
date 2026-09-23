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
BASELINE_BUDGET = 512
ERROR_TOLERANCE = 1.10


def mark_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == BASELINE_BUDGET)
        ].iloc[0]
        current = part.copy()
        current["baseline_kde_error"] = baseline["kde_abs_relative_error"]
        current["baseline_query_time_ms"] = baseline["query_time_ms"]
        current["kde_error_threshold"] = baseline["kde_abs_relative_error"] * ERROR_TOLERANCE
        current["close_density"] = current["kde_abs_relative_error"] <= current["kde_error_threshold"]
        current["faster_than_baseline"] = current["query_time_ms"] < baseline["query_time_ms"]
        current["larger_annulus"] = current["annulus_count"] > 1
        current["close_and_faster"] = (
            current["close_density"]
            & current["faster_than_baseline"]
            & current["larger_annulus"]
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "dataset":
                cells.append(str(value))
            elif column in {"annulus_count", "total_sample_budget"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_tradeoff(df: pd.DataFrame) -> None:
    datasets = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
    colors = {1: "#4c78a8", 2: "#f58518", 4: "#54a24b", 8: "#b279a2", 16: "#e45756"}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets):
        part = df[df["dataset"] == dataset]
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == BASELINE_BUDGET)
        ].iloc[0]
        threshold = baseline["kde_abs_relative_error"] * ERROR_TOLERANCE
        for annulus_count, group in part.groupby("annulus_count"):
            sizes = 34 + 1.2 * np.sqrt(group["total_sample_budget"].to_numpy(dtype=float))
            ax.scatter(
                group["query_time_ms"],
                group["kde_abs_relative_error"],
                s=sizes,
                alpha=0.82,
                color=colors[int(annulus_count)],
                edgecolor="white",
                linewidth=0.5,
                label=str(int(annulus_count)),
            )
        winners = part[part["close_and_faster"]]
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
        "Close-density and faster candidates under strict weighted total budget",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "weighted_budget_close_density_fast_tradeoff.png", dpi=260)
    fig.savefig(OUT_DIR / "weighted_budget_close_density_fast_tradeoff.pdf")
    plt.close(fig)


def main() -> None:
    df = mark_candidates(pd.read_csv(CSV_PATH))
    winners = df[df["close_and_faster"]].copy()
    winners.to_csv(IN_DIR / "weighted_budget_close_density_fast_candidates.csv", index=False)
    columns = [
        "dataset",
        "annulus_count",
        "total_sample_budget",
        "kde_sample_size",
        "kde_abs_relative_error",
        "baseline_kde_error",
        "query_time_ms",
        "baseline_query_time_ms",
        "kde_time_ms",
    ]
    summary = [
        "# Weighted Budget Close-Density Faster Candidates",
        "",
        f"Baseline is `annulus_count=1,total_sample_budget={BASELINE_BUDGET}`. ",
        f"Close density means KDE relative error <= {ERROR_TOLERANCE:.2f}x baseline error.",
        "",
    ]
    if winners.empty:
        summary.append("No larger-annulus candidate is both close-density and faster.")
    else:
        summary.append(markdown_table(winners[columns].sort_values(["dataset", "query_time_ms"]), columns))
    (IN_DIR / "weighted_budget_close_density_fast_summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )
    plot_tradeoff(df)


if __name__ == "__main__":
    main()
