from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_r25_lk_paper_angle_grid_sensitivity as base


OUT_DIR = Path("r25_l1_25_k10_100_paper_angle_grid_sensitivity")
L_VALUES = list(range(1, 26))
K_VALUES = list(range(10, 101, 10))
PREVIOUS_GRID_DIR = Path("r25_lk_paper_angle_grid_sensitivity")
PREVIOUS_L20_DIR = Path("r25_k10_200_l20_paper_angle_lookup")


def configure_base() -> None:
    base.OUT_DIR = OUT_DIR
    base.L_VALUES = L_VALUES
    base.K_VALUES = K_VALUES
    base.MAX_L = max(L_VALUES)
    base.MAX_K = max(K_VALUES)
    base.PREVIOUS_L20_DIR = PREVIOUS_L20_DIR
    base.load_seed_rows = load_seed_rows
    base.score_grid = score_grid
    base.write_outputs = write_outputs


def load_seed_rows(dataset: str, result_path: Path) -> pd.DataFrame:
    sources = []
    if result_path.exists():
        sources.append(pd.read_csv(result_path))

    previous_grid_path = PREVIOUS_GRID_DIR / dataset / "r25_lk_paper_angle_grid.csv"
    if previous_grid_path.exists():
        sources.append(pd.read_csv(previous_grid_path))

    previous_l20_path = PREVIOUS_L20_DIR / dataset / "K10_200_L20_paper_angle_lookup.csv"
    if previous_l20_path.exists():
        sources.append(pd.read_csv(previous_l20_path))

    if not sources:
        return pd.DataFrame()

    df = pd.concat(sources, ignore_index=True)
    df = df[(df["L"].isin(L_VALUES)) & (df["K"].isin(K_VALUES))].copy()
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
            0.25 * part["point_recall"]
            + 0.22 * part["kde_quality"]
            + 0.18 * part["point_f1"]
            + 0.17 * part["point_precision"]
            + 0.10 * part["size_quality"]
            + 0.05 * part["speed_quality"]
            + 0.03 * part["expansion_quality"]
        )
        part["feasible"] = (
            (part["point_recall"] >= 0.90)
            & (part["kde_abs_relative_error_vs_exact_r25"] <= 0.10)
            & (part["candidate_expansion_ratio"] <= 2.0)
        )
        scored.append(part)
    return pd.concat(scored, ignore_index=True)


def write_outputs(results: pd.DataFrame, metadata: list[dict]) -> None:
    results = score_grid(results)
    results.to_csv(OUT_DIR / "r25_l1_25_k10_100_paper_angle_grid_all.csv", index=False)

    best = base.best_by_dataset(results)
    best.to_csv(OUT_DIR / "best_lk_by_dataset.csv", index=False)

    common = base.summarize_common(results)
    common.to_csv(OUT_DIR / "common_lk_paper_angle_grid_summary.csv", index=False)

    for metric in [
        "point_recall",
        "point_precision",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "mean_hamming_threshold",
        "query_time_ms",
        "score_balanced",
    ]:
        base.pivot_metric(results, metric).to_csv(OUT_DIR / f"grid_{metric}.csv", index=False)

    base.plot_all_heatmaps(results)

    top_common = common.sort_values(
        ["mean_score_balanced", "mean_recall", "mean_kde_error", "mean_query_time_ms"],
        ascending=[False, False, True, True],
    ).head(12)
    feasible_common = common[common["feasible_all"]].sort_values(
        ["mean_score_balanced", "mean_recall", "mean_kde_error", "mean_query_time_ms"],
        ascending=[False, False, True, True],
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
        "K_step": 10,
        "angle_recall_alpha": base.ANGLE_RECALL_ALPHA,
        "sphere_layer_rule": "s=round(||p||/delta), fs=round((||q||-r)/delta), ls=round((||q||+r)/delta)",
        "theta_rule": "theta=arccos((||q||^2+||p_layer||^2-r^2)/(2||q||||p_layer||))",
        "hamming_threshold_rule": "For each query q and sphere layer s, compute theta(q,s), then I(q,s)=min I such that BinomialCDF(I; K, theta(q,s)/pi)>1-(1-alpha)^(1/L). Reported mean_hamming_threshold is only an aggregate diagnostic.",
        "lookup_rule": "retrieve bucket codes from distance tables within Hamming distance I(q,s), merge hash-table buckets, and intersect sphere layers",
        "selection": {
            "feasible": "recall >= 0.90, KDE error <= 0.10, and candidate_expansion_ratio <= 2.0",
            "score_accuracy": "0.40*recall + 0.30*kde_quality + 0.20*F1 + 0.10*precision",
            "score_balanced": "0.25*recall + 0.22*kde_quality + 0.18*F1 + 0.17*precision + 0.10*size_quality + 0.05*speed_quality + 0.03*expansion_quality",
            "note": "This pruning-aware score is intentionally less favorable to broad, weak-pruning K=10 configurations than pure F1/recall ranking.",
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
        "# R25 L=1..25, K=10..100 Paper Angle-Lookup Grid Sensitivity",
        "",
        "Query radius is query-adaptive R25 and bandwidth is h(q)=0.14R25(q).",
        "The grid varies hash tables L and hash code length K under the per-query/per-layer paper angle-to-Hamming threshold rule.",
        "",
        "## Setup",
        "",
        "- L grid: 1,2,...,25.",
        "- K grid: 10,20,...,100.",
        "- Hamming threshold: for each query q and sphere layer s, compute theta(q,s), then convert it to I(q,s) by Eq. I_threshold with Binomial(K, theta(q,s)/pi), alpha=0.95.",
        "- Feasible criterion: recall >= 0.90, KDE error <= 0.10, and candidate expansion ratio <= 2.0.",
        "- Ranking uses a pruning-aware balanced score, so broad K=10 retrieval is penalized by precision, candidate size, expansion ratio, and time.",
        "",
        "## Best L/K by Dataset",
        "",
        base.markdown_table(best, best_cols),
        "",
        "## Top Common Settings by Pruning-Aware Balanced Score",
        "",
        base.markdown_table(top_common, common_cols),
        "",
        "## Top Common Feasible Settings",
        "",
        base.markdown_table(feasible_common, common_cols),
        "",
        "## Outputs",
        "",
        "- `r25_l1_25_k10_100_paper_angle_grid_all.csv`",
        "- `common_lk_paper_angle_grid_summary.csv`",
        "- `best_lk_by_dataset.csv`",
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
