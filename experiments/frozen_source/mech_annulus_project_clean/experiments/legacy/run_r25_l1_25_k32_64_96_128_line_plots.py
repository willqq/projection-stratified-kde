from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

import run_r25_lk_paper_angle_grid_sensitivity as base
from run_r25_k10_200_l20_empirical_angle_lookup import (
    layer_max_angle_from_radius,
    paper_layer_bounds,
)


OUT_DIR = Path("r25_l1_25_k32_64_96_128_line_plots")
L_VALUES = list(range(1, 26))
K_VALUES = [32, 64, 96, 128]
BOOST_START_K = 10
FULL_BOOST_K = 50
TARGET_SINGLE_TABLE_CDF_AT_FULL_BOOST = 0.995
EXTRA_MARGIN_RATIO_AT_FULL_BOOST = 0.18
MIN_THRESHOLD_RATIO_AT_FULL_BOOST = 0.35

METRIC_SPECS = [
    ("point_recall", "Recall", "recall"),
    ("point_f1", "F1", "f1"),
    ("point_precision", "Precision", "precision"),
    ("kde_abs_relative_error_vs_exact_r25", "KDE error", "kde_error"),
    ("candidate_size", "Candidate size", "candidate_size"),
    ("mean_hamming_threshold", "Mean Hamming threshold", "hamming_threshold"),
    ("query_time_ms", "Query time (ms)", "query_time"),
]


def k_progress(bits: int) -> float:
    if FULL_BOOST_K <= BOOST_START_K:
        return 0.0
    return float(np.clip((bits - BOOST_START_K) / (FULL_BOOST_K - BOOST_START_K), 0.0, 1.0))


def recall_boosted_hamming_threshold_from_angle(
    theta: float,
    bits: int,
    tables: int,
    alpha: float,
) -> int:
    bit_flip_probability = float(np.clip(theta / math.pi, 0.0, 1.0))
    strict_target = 1.0 - (1.0 - alpha) ** (1.0 / tables)
    progress = k_progress(bits)
    boosted_target = max(
        strict_target,
        strict_target
        + progress * (TARGET_SINGLE_TABLE_CDF_AT_FULL_BOOST - strict_target),
    )
    threshold = int(binom.ppf(boosted_target, bits, bit_flip_probability))
    threshold += int(math.ceil(EXTRA_MARGIN_RATIO_AT_FULL_BOOST * progress * bits))

    min_ratio = MIN_THRESHOLD_RATIO_AT_FULL_BOOST * progress
    threshold = max(threshold, int(math.ceil(min_ratio * bits)))
    return min(max(threshold, 0), bits)


def configure_base() -> None:
    base.OUT_DIR = OUT_DIR
    base.L_VALUES = L_VALUES
    base.K_VALUES = K_VALUES
    base.MAX_L = max(L_VALUES)
    base.MAX_K = max(K_VALUES)
    base.hamming_threshold_from_angle = recall_boosted_hamming_threshold_from_angle
    base.paper_layer_bounds = paper_layer_bounds
    base.layer_angle_from_radius = layer_max_angle_from_radius
    base.write_outputs = write_outputs


def score_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kde_quality"] = (
        1.0 - df["kde_abs_relative_error_vs_exact_r25"]
    ).clip(lower=0.0, upper=1.0)
    scored = []
    for dataset, part in df.groupby("dataset"):
        part = part.copy()
        time_max = max(float(part["query_time_ms"].max()), 1e-12)
        size_max = max(float(part["candidate_size"].max()), 1e-12)
        part["speed_quality"] = (1.0 - part["query_time_ms"] / time_max).clip(0.0, 1.0)
        part["size_quality"] = (1.0 - part["candidate_size"] / size_max).clip(0.0, 1.0)
        part["score_f1_objective"] = part["point_f1"]
        part["score_accuracy"] = (
            0.40 * part["point_recall"]
            + 0.30 * part["kde_quality"]
            + 0.20 * part["point_f1"]
            + 0.10 * part["point_precision"]
        )
        part["score_balanced"] = (
            0.30 * part["point_recall"]
            + 0.25 * part["kde_quality"]
            + 0.25 * part["point_f1"]
            + 0.10 * part["point_precision"]
            + 0.05 * part["size_quality"]
            + 0.05 * part["speed_quality"]
        )
        part["feasible"] = (
            (part["point_recall"] >= 0.90)
            & (part["kde_abs_relative_error_vs_exact_r25"] <= 0.10)
        )
        scored.append(part)
    return pd.concat(scored, ignore_index=True)


def plot_metric_by_l(
    frame: pd.DataFrame,
    out_dir: Path,
    metric: str,
    title: str,
    filename: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759"]
    for color, k_value in zip(colors, K_VALUES):
        part = frame[frame["K"] == k_value].sort_values("L")
        ax.plot(
            part["L"],
            part[metric],
            marker="o",
            linewidth=1.7,
            markersize=3.2,
            color=color,
            label=f"K={k_value}",
        )
    ax.set_title(title)
    ax.set_xlabel("Hash tables L")
    ax.set_ylabel(title.split(": ")[-1])
    ax.set_xticks(L_VALUES)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename}.png", dpi=450)
    fig.savefig(out_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_dataset_metric_lines(results: pd.DataFrame) -> None:
    for dataset, part in results.groupby("dataset"):
        figure_dir = OUT_DIR / "figures" / "by_dataset" / dataset
        for metric, label, filename in METRIC_SPECS:
            plot_metric_by_l(
                part,
                figure_dir,
                metric,
                f"{dataset}: {label} by hash tables",
                f"{filename}_by_l_k_lines",
            )

    aggregate = (
        results.groupby(["K", "L"], as_index=False)
        .agg({metric: "mean" for metric, _, _ in METRIC_SPECS})
    )
    aggregate_dir = OUT_DIR / "figures" / "aggregate"
    for metric, label, filename in METRIC_SPECS:
        plot_metric_by_l(
            aggregate,
            aggregate_dir,
            metric,
            f"Mean: {label} by hash tables",
            f"mean_{filename}_by_l_k_lines",
        )


def plot_per_k_dashboards(results: pd.DataFrame) -> None:
    dashboard_metrics = [
        ("point_recall", "Recall"),
        ("point_f1", "F1"),
        ("point_precision", "Precision"),
        ("candidate_size", "Candidate size"),
    ]
    for k_value in K_VALUES:
        figure_dir = OUT_DIR / "figures" / "by_k" / f"K{k_value}"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for dataset, part in results[results["K"] == k_value].groupby("dataset"):
            part = part.sort_values("L")
            fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
            for ax, (metric, label) in zip(axes.ravel(), dashboard_metrics):
                ax.plot(
                    part["L"],
                    part[metric],
                    marker="o",
                    linewidth=1.6,
                    markersize=3.0,
                    color="#4c78a8",
                )
                ax.set_title(label)
                ax.set_xlabel("Hash tables L")
                ax.grid(True, alpha=0.25)
            fig.suptitle(f"{dataset}: K={k_value} metrics by hash tables", y=0.99)
            fig.tight_layout()
            fig.savefig(figure_dir / f"{dataset}_K{k_value}_metrics_by_l.png", dpi=450)
            fig.savefig(figure_dir / f"{dataset}_K{k_value}_metrics_by_l.pdf")
            plt.close(fig)


def best_by_dataset_f1(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in results.groupby("dataset"):
        best = part.sort_values(
            [
                "point_f1",
                "kde_abs_relative_error_vs_exact_r25",
                "point_recall",
                "candidate_size",
                "query_time_ms",
            ],
            ascending=[False, True, False, True, True],
        ).iloc[0].copy()
        best["selection"] = "f1_objective"
        rows.append(best)
    return pd.DataFrame(rows)


def best_by_dataset_recall(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in results.groupby("dataset"):
        best = part.sort_values(
            [
                "point_recall",
                "point_f1",
                "kde_abs_relative_error_vs_exact_r25",
                "candidate_size",
                "query_time_ms",
            ],
            ascending=[False, False, True, True, True],
        ).iloc[0].copy()
        best["selection"] = "recall_objective"
        rows.append(best)
    return pd.DataFrame(rows)


def write_outputs(results: pd.DataFrame, metadata: list[dict]) -> None:
    results = score_grid(results)
    results.to_csv(OUT_DIR / "r25_l1_25_k32_64_96_128_line_grid_all.csv", index=False)

    best_f1 = best_by_dataset_f1(results)
    best_f1.to_csv(OUT_DIR / "best_f1_by_dataset.csv", index=False)
    best_recall = best_by_dataset_recall(results)
    best_recall.to_csv(OUT_DIR / "best_recall_by_dataset.csv", index=False)

    common = base.summarize_common(results)
    common.to_csv(OUT_DIR / "common_k32_64_96_128_l1_25_summary.csv", index=False)

    for metric, _, _ in METRIC_SPECS:
        base.pivot_metric(results, metric).to_csv(OUT_DIR / f"grid_{metric}.csv", index=False)

    plot_dataset_metric_lines(results)
    plot_per_k_dashboards(results)

    best_cols = [
        "dataset",
        "selection",
        "L",
        "K",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "mean_hamming_threshold",
        "query_time_ms",
    ]
    top_common = common.sort_values(
        ["mean_f1", "mean_recall", "mean_kde_error", "mean_candidate_size"],
        ascending=[False, False, True, True],
    ).head(12)
    common_cols = [
        "L",
        "K",
        "mean_precision",
        "mean_recall",
        "min_recall",
        "mean_f1",
        "mean_kde_error",
        "mean_candidate_size",
        "mean_hamming_threshold",
        "mean_query_time_ms",
    ]

    config = {
        "datasets": base.DATASETS,
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": 25.0,
        "bandwidth_mode": "h(q)=0.14R25(q)",
        "bandwidth_factor": base.BANDWIDTH_FACTOR,
        "delta_mode": "delta=0.25*median_q R25(q)",
        "norm_filter_rule": base.NORM_FILTER_RULE,
        "lookup_rule": base.LOOKUP_RULE,
        "angle_recall_alpha": base.ANGLE_RECALL_ALPHA,
        "boost_start_k": BOOST_START_K,
        "full_boost_k": FULL_BOOST_K,
        "target_single_table_cdf_at_full_boost": TARGET_SINGLE_TABLE_CDF_AT_FULL_BOOST,
        "extra_margin_ratio_at_full_boost": EXTRA_MARGIN_RATIO_AT_FULL_BOOST,
        "min_hamming_threshold_ratio_at_full_boost": MIN_THRESHOLD_RATIO_AT_FULL_BOOST,
        "line_plot_rule": "x-axis is hash tables L; curves are K=32,64,96,128.",
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = [
        "# R25 K=32/64/96/128 Line Plots by Hash Tables",
        "",
        "This run evaluates L=1..25 for K=32,64,96,128 under query-adaptive R25.",
        f"Norm-table filter: `{base.NORM_FILTER_RULE}`.",
        "",
        "## Best F1 by Dataset",
        "",
        base.markdown_table(best_f1, best_cols),
        "",
        "## Best Recall by Dataset",
        "",
        base.markdown_table(best_recall, best_cols),
        "",
        "## Top Common Settings by Mean F1",
        "",
        base.markdown_table(top_common, common_cols),
        "",
        "## Outputs",
        "",
        "- `r25_l1_25_k32_64_96_128_line_grid_all.csv`",
        "- `common_k32_64_96_128_l1_25_summary.csv`",
        "- `best_f1_by_dataset.csv`",
        "- `best_recall_by_dataset.csv`",
        "- `figures/by_dataset/<dataset>/<metric>_by_l_k_lines.png`",
        "- `figures/aggregate/mean_<metric>_by_l_k_lines.png`",
        "- `figures/by_k/K{32,64,96,128}/<dataset>_K*_metrics_by_l.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
