from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

import run_r25_lk_paper_angle_grid_sensitivity as base


OUT_DIR = Path("r25_l1_25_k10_100_recall_boosted_angle_grid_sensitivity")
L_VALUES = list(range(1, 26))
K_VALUES = list(range(10, 101, 10))
PREVIOUS_STRICT_GRID_DIR = Path("r25_l1_25_k10_100_paper_angle_grid_sensitivity")
PREVIOUS_GRID_DIR = Path("r25_lk_paper_angle_grid_sensitivity")
PREVIOUS_L20_DIR = Path("r25_k10_200_l20_paper_angle_lookup")

TARGET_SINGLE_TABLE_CDF_AT_MAX_K = 0.90
EXTRA_MARGIN_RATIO_AT_MAX_K = 0.04
MIN_THRESHOLD_RATIO_AT_MAX_K = 0.0
FULL_BOOST_K = max(K_VALUES)
RECALL_TARGET_K = 50


def k_progress(bits: int) -> float:
    boost_start_k = min(K_VALUES)
    if FULL_BOOST_K <= boost_start_k:
        return 0.0
    return float(np.clip((bits - boost_start_k) / (FULL_BOOST_K - boost_start_k), 0.0, 1.0))


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
        + progress * (TARGET_SINGLE_TABLE_CDF_AT_MAX_K - strict_target),
    )
    threshold = int(binom.ppf(boosted_target, bits, bit_flip_probability))
    threshold += int(math.ceil(EXTRA_MARGIN_RATIO_AT_MAX_K * progress * bits))

    min_ratio = MIN_THRESHOLD_RATIO_AT_MAX_K * progress
    threshold = max(threshold, int(math.ceil(min_ratio * bits)))
    return min(max(threshold, 0), bits)


def configure_base() -> None:
    base.OUT_DIR = OUT_DIR
    base.L_VALUES = L_VALUES
    base.K_VALUES = K_VALUES
    base.MAX_L = max(L_VALUES)
    base.MAX_K = max(K_VALUES)
    base.PREVIOUS_L20_DIR = PREVIOUS_L20_DIR
    base.hamming_threshold_from_angle = recall_boosted_hamming_threshold_from_angle
    base.load_seed_rows = load_seed_rows
    base.score_grid = score_grid
    base.write_outputs = write_outputs


def load_seed_rows(dataset: str, result_path: Path) -> pd.DataFrame:
    sources = []
    if result_path.exists():
        sources.append(pd.read_csv(result_path))

    # Only reuse rows already produced by this boosted experiment. Strict-threshold
    # rows are intentionally not reused because the retrieval rule changed.
    if not sources:
        return pd.DataFrame()

    df = pd.concat(sources, ignore_index=True)
    df = df[(df["L"].isin(L_VALUES)) & (df["K"].isin(K_VALUES))].copy()
    if df.empty:
        return df
    if "norm_filter_rule" not in df.columns:
        return pd.DataFrame()
    df = df[df["norm_filter_rule"] == base.NORM_FILTER_RULE].copy()
    if df.empty:
        return df
    return df.sort_values(["dataset", "L", "K"]).drop_duplicates(
        ["dataset", "L", "K"],
        keep="last",
    )


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
        part["expansion_quality"] = (
            1.0 / (1.0 + np.maximum(part["candidate_expansion_ratio"] - 1.0, 0.0))
        )
        part["score_accuracy"] = (
            0.40 * part["point_recall"]
            + 0.30 * part["kde_quality"]
            + 0.20 * part["point_f1"]
            + 0.10 * part["point_precision"]
        )
        part["score_balanced"] = (
            0.24 * part["point_recall"]
            + 0.21 * part["kde_quality"]
            + 0.18 * part["point_f1"]
            + 0.17 * part["point_precision"]
            + 0.10 * part["size_quality"]
            + 0.05 * part["speed_quality"]
            + 0.03 * part["expansion_quality"]
            + 0.02 * np.clip((part["K"] - min(K_VALUES)) / (max(K_VALUES) - min(K_VALUES)), 0.0, 1.0)
        )
        part["score_f1_objective"] = part["point_f1"]
        part["feasible"] = (
            (part["point_recall"] >= 0.90)
            & (part["kde_abs_relative_error_vs_exact_r25"] <= 0.10)
            & (part["candidate_expansion_ratio"] <= 2.0)
        )
        scored.append(part)
    return pd.concat(scored, ignore_index=True)


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


def best_by_dataset_recall_k50_preferred(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in results.groupby("dataset"):
        ranked = part.copy()
        ranked["target_k_distance"] = (ranked["K"] - RECALL_TARGET_K).abs()
        best = ranked.sort_values(
            [
                "point_recall",
                "target_k_distance",
                "point_f1",
                "kde_abs_relative_error_vs_exact_r25",
                "candidate_size",
                "query_time_ms",
            ],
            ascending=[False, True, False, True, True, True],
        ).iloc[0].copy()
        best["selection"] = "recall_k50_preferred"
        rows.append(best.drop(labels=["target_k_distance"]))
    return pd.DataFrame(rows)


def write_outputs(results: pd.DataFrame, metadata: list[dict]) -> None:
    results = score_grid(results)
    results.to_csv(OUT_DIR / "r25_l1_25_k10_100_recall_boosted_angle_grid_all.csv", index=False)

    best = best_by_dataset_f1(results)
    best.to_csv(OUT_DIR / "best_lk_by_dataset.csv", index=False)
    best_recall_k50 = best_by_dataset_recall_k50_preferred(results)
    best_recall_k50.to_csv(OUT_DIR / "best_recall_k50_preferred_by_dataset.csv", index=False)

    common = base.summarize_common(results)
    common.to_csv(OUT_DIR / "common_lk_recall_boosted_angle_grid_summary.csv", index=False)

    for metric in [
        "point_recall",
        "point_precision",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "mean_hamming_threshold",
        "query_time_ms",
        "score_balanced",
        "score_f1_objective",
    ]:
        base.pivot_metric(results, metric).to_csv(OUT_DIR / f"grid_{metric}.csv", index=False)

    base.plot_all_heatmaps(results)

    top_common = common.sort_values(
        ["mean_f1", "mean_kde_error", "mean_recall", "mean_candidate_size", "mean_query_time_ms"],
        ascending=[False, True, False, True, True],
    ).head(12)
    feasible_common = common[common["feasible_all"]].sort_values(
        ["mean_f1", "mean_kde_error", "mean_recall", "mean_candidate_size", "mean_query_time_ms"],
        ascending=[False, True, False, True, True],
    ).head(12)

    config = {
        "datasets": base.DATASETS,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": 25.0,
        "bandwidth_mode": "h(q)=0.14R25(q)",
        "bandwidth_factor": base.BANDWIDTH_FACTOR,
        "delta_mode": "delta=0.25*median_q R25(q)",
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "angle_recall_alpha": base.ANGLE_RECALL_ALPHA,
        "target_single_table_cdf_at_max_k": TARGET_SINGLE_TABLE_CDF_AT_MAX_K,
        "extra_margin_ratio_at_max_k": EXTRA_MARGIN_RATIO_AT_MAX_K,
        "min_hamming_threshold_ratio_at_max_k": MIN_THRESHOLD_RATIO_AT_MAX_K,
        "full_boost_k": FULL_BOOST_K,
        "norm_filter_rule": base.NORM_FILTER_RULE,
        "lookup_rule": base.LOOKUP_RULE,
        "hamming_threshold_rule": "For each query q and sphere layer s, compute theta(q,s). Start from Eq. I_threshold, then increase the single-table Binomial CDF target and add a small margin as K grows, so larger K settings are less aggressively pruned.",
        "selection": {
            "primary_objective": "maximize point_f1; ties prefer lower KDE error, higher recall, smaller candidate size, and lower query time",
            "recall_k50_preferred": "maximize point_recall; ties prefer K=50, then higher F1, lower KDE error, smaller candidate size, and lower query time",
            "feasible": "recall >= 0.90, KDE error <= 0.10, and candidate_expansion_ratio <= 2.0",
            "score_f1_objective": "point_f1",
            "score_accuracy": "0.40*recall + 0.30*kde_quality + 0.20*F1 + 0.10*precision",
            "score_balanced": "0.24*recall + 0.21*kde_quality + 0.18*F1 + 0.17*precision + 0.10*size_quality + 0.05*speed_quality + 0.03*expansion_quality + 0.02*K_progress",
        },
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
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "candidate_expansion_ratio",
        "mean_hamming_threshold",
        "query_time_ms",
        "score_f1_objective",
        "score_balanced",
        "score_accuracy",
        "feasible",
    ]
    common_cols = [
        "L",
        "K",
        "mean_precision",
        "mean_recall",
        "min_recall",
        "mean_f1",
        "mean_kde_error",
        "max_kde_error",
        "mean_candidate_size",
        "mean_hamming_threshold",
        "mean_query_time_ms",
        "mean_score_balanced",
        "feasible_all",
    ]

    summary = [
        "# R25 L=1..25, K=10..100 F1-Optimized Angle Grid Sensitivity",
        "",
        "This run keeps the per-query/per-layer angle-to-Hamming conversion, but relaxes the Hamming threshold as K grows.",
        "The primary model-selection objective is point F1; ties prefer lower KDE error, higher recall, smaller candidate size, and lower query time.",
        f"A separate recall report maximizes recall and resolves ties toward K={RECALL_TARGET_K}.",
        "",
        "## Setup",
        "",
        "- L grid: 1,2,...,25.",
        "- K grid: 10,20,...,100.",
        "- Radius: query-adaptive R25.",
        "- Bandwidth: h(q)=0.14R25(q).",
        f"- Norm-table filter: {base.NORM_FILTER_RULE}.",
        f"- Single-table Binomial CDF target is gradually boosted up to {TARGET_SINGLE_TABLE_CDF_AT_MAX_K:.3g} by K={FULL_BOOST_K}.",
        f"- Extra Hamming margin grows up to {EXTRA_MARGIN_RATIO_AT_MAX_K:.3g}K by K={FULL_BOOST_K}.",
        f"- Minimum Hamming threshold grows up to {MIN_THRESHOLD_RATIO_AT_MAX_K:.3g}K by K={FULL_BOOST_K}.",
        "",
        "## Best L/K by Dataset",
        "",
        base.markdown_table(best, best_cols),
        "",
        f"## Best Recall by Dataset, K={RECALL_TARGET_K} Preferred",
        "",
        base.markdown_table(best_recall_k50, best_cols),
        "",
        "## Top Common Settings by Mean F1",
        "",
        base.markdown_table(top_common, common_cols),
        "",
        "## Top Common Feasible Settings",
        "",
        base.markdown_table(feasible_common, common_cols),
        "",
        "## Outputs",
        "",
        "- `r25_l1_25_k10_100_recall_boosted_angle_grid_all.csv`",
        "- `common_lk_recall_boosted_angle_grid_summary.csv`",
        "- `best_lk_by_dataset.csv`",
        "- `best_recall_k50_preferred_by_dataset.csv`",
        "- `figures/<dataset>/recall_heatmap.png`",
        "- `figures/<dataset>/kde_error_heatmap.png`",
        "- `figures/<dataset>/balanced_score_heatmap.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
