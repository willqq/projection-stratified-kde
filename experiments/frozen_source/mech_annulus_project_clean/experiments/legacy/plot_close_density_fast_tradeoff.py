from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_DIR = Path("mech_annulus_count_total_sample_sensitivity")
OUT_DIR = IN_DIR / "figures"
CSV_PATH = IN_DIR / "all_annulus_count_total_sample.csv"
BASELINE_BUDGET = 512
ERROR_TOLERANCE = 1.10


def mark_candidates(df: pd.DataFrame, time_metric: str) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == BASELINE_BUDGET)
        ].iloc[0]
        threshold = baseline["kde_abs_relative_error"] * ERROR_TOLERANCE
        current = part.copy()
        current["baseline_kde_error"] = baseline["kde_abs_relative_error"]
        current["baseline_time"] = baseline[time_metric]
        current["kde_error_threshold"] = threshold
        current["close_density"] = current["kde_abs_relative_error"] <= threshold
        current["faster_than_baseline"] = current[time_metric] < baseline[time_metric]
        current["larger_annulus"] = current["annulus_count"] > 1
        current["close_and_faster"] = (
            current["larger_annulus"]
            & current["close_density"]
            & current["faster_than_baseline"]
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def plot_tradeoff(df: pd.DataFrame, time_metric: str, out_name: str, title: str) -> None:
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
            sizes = 25 + 2.4 * np.sqrt(group["kde_sample_size"].to_numpy(dtype=float))
            ax.scatter(
                group[time_metric],
                group["kde_abs_relative_error"],
                s=sizes,
                alpha=0.82,
                color=colors[int(annulus_count)],
                label=str(int(annulus_count)),
                edgecolor="white",
                linewidth=0.5,
            )
        winners = part[part["close_and_faster"]]
        if not winners.empty:
            ax.scatter(
                winners[time_metric],
                winners["kde_abs_relative_error"],
                s=170,
                facecolor="none",
                edgecolor="black",
                linewidth=1.4,
            )
        ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1.0)
        ax.axvline(baseline[time_metric], color="#333333", linestyle=":", linewidth=1.0)
        ax.scatter(
            [baseline[time_metric]],
            [baseline["kde_abs_relative_error"]],
            marker="*",
            s=180,
            color="black",
            label="baseline",
            zorder=5,
        )
        ax.set_title(dataset)
        ax.set_xlabel(time_metric.replace("_", " "))
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
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
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
            if column in {"dataset"}:
                cells.append(str(value))
            elif column in {"annulus_count", "total_sample_budget", "per_ring_sample_size"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    total_df = mark_candidates(df, "query_time_ms")
    kde_df = mark_candidates(df, "kde_time_ms")

    total_candidates = total_df[total_df["close_and_faster"]].copy()
    kde_candidates = kde_df[kde_df["close_and_faster"]].copy()
    total_candidates.to_csv(IN_DIR / "close_density_faster_total_query_candidates.csv", index=False)
    kde_candidates.to_csv(IN_DIR / "close_density_faster_kde_stage_candidates.csv", index=False)

    plot_tradeoff(
        total_df,
        "query_time_ms",
        "close_density_total_query_time_tradeoff",
        "Close-density and faster total-query candidates",
    )
    plot_tradeoff(
        kde_df,
        "kde_time_ms",
        "close_density_kde_stage_time_tradeoff",
        "Close-density and faster KDE-stage candidates",
    )

    summary_cols = [
        "dataset",
        "annulus_count",
        "total_sample_budget",
        "per_ring_sample_size",
        "kde_sample_size",
        "kde_abs_relative_error",
        "baseline_kde_error",
        "query_time_ms",
        "kde_time_ms",
    ]
    with (IN_DIR / "close_density_fast_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Close Density And Faster Candidates\n\n")
        f.write(
            f"Baseline is `annulus_count=1,total_sample_budget={BASELINE_BUDGET}`. "
            f"Close density means KDE relative error <= {ERROR_TOLERANCE:.2f}x baseline error.\n\n"
        )
        f.write("## Faster Total Query Time\n\n")
        if total_candidates.empty:
            f.write("No larger-annulus candidate is both close-density and faster in total query time.\n\n")
        else:
            f.write(
                markdown_table(
                    total_candidates[summary_cols].sort_values(["dataset", "query_time_ms"]),
                    summary_cols,
                )
            )
            f.write("\n\n")
        f.write("## Faster KDE Stage Time\n\n")
        if kde_candidates.empty:
            f.write("No larger-annulus candidate is both close-density and faster in KDE-stage time.\n")
        else:
            f.write(
                markdown_table(
                    kde_candidates[summary_cols].sort_values(["dataset", "kde_time_ms"]),
                    summary_cols,
                )
            )
            f.write("\n")


if __name__ == "__main__":
    main()
