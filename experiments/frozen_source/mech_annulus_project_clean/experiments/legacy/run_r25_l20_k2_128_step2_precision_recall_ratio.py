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


OUT_DIR = Path("r25_l20_k2_128_no_norm_filter_recall_tuned")
L_VALUES = [20]
K_VALUES = list(range(2, 129, 2))
BOOST_START_K = 10
FULL_BOOST_K = 50
BOOST_END_K = max(K_VALUES)
TARGET_SINGLE_TABLE_CDF_AT_FULL_BOOST = 0.995
EXTRA_MARGIN_RATIO_AT_FULL_BOOST = 0.18
MIN_THRESHOLD_RATIO_AT_FULL_BOOST = 0.35
RATIO_METRIC = "candidate_expansion_ratio"
RATIO_LABEL = "Candidate/true ratio"
NO_NORM_FILTER_RULE = "disabled: no norm-table or radial-layer candidate filter"
NO_NORM_LOOKUP_RULE = (
    "distance-table hash-bucket lookup with per-layer thresholds; "
    "union over L tables without intersecting sphere-layer masks or exact norm-table filter"
)


def k_progress(bits: int) -> float:
    if FULL_BOOST_K <= BOOST_START_K:
        return 0.0
    if bits <= FULL_BOOST_K:
        return float(np.clip((bits - BOOST_START_K) / (FULL_BOOST_K - BOOST_START_K), 0.0, 1.0))
    if BOOST_END_K <= FULL_BOOST_K:
        return 1.0
    return float(np.clip((BOOST_END_K - bits) / (BOOST_END_K - FULL_BOOST_K), 0.0, 1.0))


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


def no_norm_filter_ball_query_lk(
    lookups: list,
    query_bits: list[np.ndarray],
    q_id: int,
    q_norm: float,
    radius: float,
    sphere_table: dict[int, np.ndarray],
    layer_masks: dict[int, np.ndarray],
    norm_table,
    delta: float,
    n_index: int,
    table_count: int,
) -> tuple[np.ndarray, float, float]:
    low, high = base.paper_layer_bounds(q_norm, radius, delta)
    selected = np.zeros(n_index, dtype=bool)
    thresholds = []
    theta_values = []

    for layer in range(low, high + 1):
        layer_ids = sphere_table.get(layer)
        if layer_ids is None or len(layer_ids) == 0:
            continue

        theta = base.layer_angle_from_radius(q_norm, layer, delta, radius)
        threshold = base.hamming_threshold_from_angle(
            theta,
            lookups[0].bits,
            table_count,
            base.ANGLE_RECALL_ALPHA,
        )
        thresholds.append(threshold)
        theta_values.append(theta)

        for table_id in range(table_count):
            ids = lookups[table_id].ids_within_hamming(
                query_bits[table_id][q_id],
                threshold,
            )
            if len(ids):
                selected[ids] = True

    return (
        np.flatnonzero(selected).astype(np.int32),
        float(np.mean(thresholds)) if thresholds else 0.0,
        float(np.mean(theta_values)) if theta_values else 0.0,
    )


def load_seed_rows(dataset: str, result_path: Path) -> pd.DataFrame:
    if not result_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(result_path)
    if "norm_filter_rule" not in df.columns:
        return pd.DataFrame()
    df = df[df["norm_filter_rule"] == base.NORM_FILTER_RULE].copy()
    df = df[(df["L"].isin(L_VALUES)) & (df["K"].isin(K_VALUES))].copy()
    return df.sort_values(["dataset", "L", "K"]).drop_duplicates(
        ["dataset", "L", "K"],
        keep="last",
    )


def configure_base() -> None:
    base.OUT_DIR = OUT_DIR
    base.L_VALUES = L_VALUES
    base.K_VALUES = K_VALUES
    base.MAX_L = max(L_VALUES)
    base.MAX_K = max(K_VALUES)
    base.NORM_FILTER_RULE = NO_NORM_FILTER_RULE
    base.LOOKUP_RULE = NO_NORM_LOOKUP_RULE
    base.hamming_threshold_from_angle = recall_boosted_hamming_threshold_from_angle
    base.paper_layer_bounds = paper_layer_bounds
    base.layer_angle_from_radius = layer_max_angle_from_radius
    base.paper_ball_query_lk = no_norm_filter_ball_query_lk
    base.load_seed_rows = load_seed_rows
    base.write_outputs = write_outputs


def score_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["candidate_to_true_ratio"] = df[RATIO_METRIC]
    df["kde_quality"] = (
        1.0 - df["kde_abs_relative_error_vs_exact_r25"]
    ).clip(lower=0.0, upper=1.0)
    scored = []
    for dataset, part in df.groupby("dataset"):
        part = part.copy()
        time_max = max(float(part["query_time_ms"].max()), 1e-12)
        ratio_max = max(float(part["candidate_to_true_ratio"].max()), 1e-12)
        part["speed_quality"] = (1.0 - part["query_time_ms"] / time_max).clip(0.0, 1.0)
        part["ratio_quality"] = (1.0 - part["candidate_to_true_ratio"] / ratio_max).clip(0.0, 1.0)
        part["score_precision_recall_ratio"] = (
            0.35 * part["point_f1"]
            + 0.25 * part["point_recall"]
            + 0.25 * part["point_precision"]
            + 0.10 * part["ratio_quality"]
            + 0.05 * part["speed_quality"]
        )
        part["feasible"] = (
            (part["point_recall"] >= 0.90)
            & (part["candidate_to_true_ratio"] <= 2.0)
        )
        scored.append(part)
    return pd.concat(scored, ignore_index=True)


def plot_dataset_curves(results: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures" / "by_dataset"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    for dataset, part in results.groupby("dataset"):
        part = part.sort_values("K")

        fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
        axes[0].plot(part["K"], part["point_precision"], linewidth=1.7, label="Precision")
        axes[0].plot(part["K"], part["point_recall"], linewidth=1.7, label="Recall")
        axes[0].plot(part["K"], part["point_f1"], linewidth=1.5, label="F1")
        axes[0].set_title(f"{dataset}: precision/recall at L=20")
        axes[0].set_ylabel("Metric value")
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False, ncol=3)

        axes[1].plot(
            part["K"],
            part["candidate_to_true_ratio"],
            linewidth=1.7,
            color="#e15759",
            label=RATIO_LABEL,
        )
        axes[1].set_title(f"{dataset}: candidate/true ratio at L=20")
        axes[1].set_xlabel("Hash code length K")
        axes[1].set_ylabel(RATIO_LABEL)
        axes[1].set_xticks(K_VALUES[::4])
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False)

        fig.tight_layout()
        fig.savefig(figure_dir / f"{dataset}_L20_K2_128_precision_recall_ratio.png", dpi=450)
        fig.savefig(figure_dir / f"{dataset}_L20_K2_128_precision_recall_ratio.pdf")
        plt.close(fig)


def plot_aggregate_curves(results: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures" / "aggregate"
    figure_dir.mkdir(parents=True, exist_ok=True)
    aggregate = (
        results.groupby("K", as_index=False)
        .agg(
            mean_precision=("point_precision", "mean"),
            mean_recall=("point_recall", "mean"),
            mean_f1=("point_f1", "mean"),
            mean_candidate_to_true_ratio=("candidate_to_true_ratio", "mean"),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    axes[0].plot(aggregate["K"], aggregate["mean_precision"], linewidth=1.8, label="Mean precision")
    axes[0].plot(aggregate["K"], aggregate["mean_recall"], linewidth=1.8, label="Mean recall")
    axes[0].plot(aggregate["K"], aggregate["mean_f1"], linewidth=1.5, label="Mean F1")
    axes[0].set_title("Mean precision/recall at L=20")
    axes[0].set_ylabel("Metric value")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(
        aggregate["K"],
        aggregate["mean_candidate_to_true_ratio"],
        linewidth=1.8,
        color="#e15759",
        label=f"Mean {RATIO_LABEL}",
    )
    axes[1].set_title("Mean candidate/true ratio at L=20")
    axes[1].set_xlabel("Hash code length K")
    axes[1].set_ylabel(RATIO_LABEL)
    axes[1].set_xticks(K_VALUES[::4])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(figure_dir / "mean_L20_K2_128_precision_recall_ratio.png", dpi=450)
    fig.savefig(figure_dir / "mean_L20_K2_128_precision_recall_ratio.pdf")
    plt.close(fig)
    aggregate.to_csv(OUT_DIR / "mean_L20_K2_128_precision_recall_ratio.csv", index=False)


def best_by_dataset(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in results.groupby("dataset"):
        best = part.sort_values(
            [
                "score_precision_recall_ratio",
                "point_f1",
                "point_recall",
                "candidate_to_true_ratio",
                "query_time_ms",
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0].copy()
        best["selection"] = "precision_recall_ratio_score"
        rows.append(best)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col in {"dataset", "selection"}:
                cells.append(str(value))
            elif col in {"L", "K"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(results: pd.DataFrame, metadata: list[dict]) -> None:
    results = score_grid(results)
    results.to_csv(OUT_DIR / "r25_l20_k2_128_no_norm_filter_recall_tuned_all.csv", index=False)

    l20_table = results[
        [
            "dataset",
            "L",
            "K",
            "point_precision",
            "point_recall",
            "point_f1",
            "candidate_size",
            "exact_r25_candidate_size",
            "candidate_to_true_ratio",
            "candidate_expansion_ratio",
            "query_time_ms",
            "mean_hamming_threshold",
        ]
    ].sort_values(["dataset", "K"])
    l20_table.to_csv(OUT_DIR / "L20_K2_128_no_norm_filter_recall_tuned.csv", index=False)

    best = best_by_dataset(results)
    best.to_csv(OUT_DIR / "best_no_norm_filter_recall_tuned_by_dataset.csv", index=False)

    plot_dataset_curves(results)
    plot_aggregate_curves(results)

    config = {
        "datasets": base.DATASETS,
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "radius_mode": "query_adaptive_R25",
        "bandwidth_mode": "h(q)=0.14R25(q)",
        "norm_filter_rule": base.NORM_FILTER_RULE,
        "lookup_rule": base.LOOKUP_RULE,
        "candidate_to_true_ratio": "|approximate candidate set| / |exact R25 true set|",
        "boost_start_k": BOOST_START_K,
        "full_boost_k": FULL_BOOST_K,
        "boost_end_k": BOOST_END_K,
        "boost_schedule": "recall boost increases from K=10 to K=50, then decreases toward K=128",
        "target_single_table_cdf_at_full_boost": TARGET_SINGLE_TABLE_CDF_AT_FULL_BOOST,
        "extra_margin_ratio_at_full_boost": EXTRA_MARGIN_RATIO_AT_FULL_BOOST,
        "min_hamming_threshold_ratio_at_full_boost": MIN_THRESHOLD_RATIO_AT_FULL_BOOST,
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_cols = [
        "dataset",
        "selection",
        "L",
        "K",
        "point_precision",
        "point_recall",
        "point_f1",
        "candidate_to_true_ratio",
        "candidate_size",
        "exact_r25_candidate_size",
        "query_time_ms",
        "score_precision_recall_ratio",
    ]
    summary = [
        "# R25 L=20, K=2..128 No-Norm-Filter Recall Tuning",
        "",
        "Fixed hash tables L=20 and varied hash code length K=2,4,...,128.",
        "The original norm-table and radial-layer candidate filters are disabled; layers are only used to compute per-layer angular Hamming thresholds.",
        "The recall boost increases until K=50 and then decreases toward K=128 so the recall-focused setting is centered around K=50.",
        f"Candidate/true ratio is `{RATIO_METRIC}` = |candidate set| / |exact R25 true set|.",
        f"Norm/radial filter: `{base.NORM_FILTER_RULE}`.",
        "",
        "## Best Settings by Precision/Recall/Ratio Score",
        "",
        markdown_table(best, best_cols),
        "",
        "## Outputs",
        "",
        "- `r25_l20_k2_128_no_norm_filter_recall_tuned_all.csv`",
        "- `L20_K2_128_no_norm_filter_recall_tuned.csv`",
        "- `mean_L20_K2_128_precision_recall_ratio.csv`",
        "- `best_no_norm_filter_recall_tuned_by_dataset.csv`",
        "- `figures/by_dataset/<dataset>_L20_K2_128_precision_recall_ratio.png`",
        "- `figures/aggregate/mean_L20_K2_128_precision_recall_ratio.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
