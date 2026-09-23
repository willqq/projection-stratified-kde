from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import mech_annulus_experiments as exp
import run_annulus_count_total_sample_sensitivity as base


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("mech_annulus_count_weighted_budget_sensitivity")

SPLIT = {"n_train": 2800, "n_index": 1800, "n_queries": 40}
MAX_TABLES = 18
MAX_BITS = 12
COMMON = {
    "L": 12,
    "K": 5,
    "radius_percentile": 70,
    "bandwidth_factor": 0.25,
    "hamming_probe": 2,
    "min_collisions": 2,
}

NORM_PERCENTILE = 25
DELTA_DIVISIONS = 12
ANNULUS_COUNTS = [1, 2, 4, 8, 16]
TOTAL_SAMPLE_BUDGETS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
ALLOCATION_MODE = "size_weighted_total_budget"


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, dict]:
    exp.set_seed(exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    x = exp.load_dataset(dataset)
    sizes = base.split_sizes(len(x))
    train_x, index_x, queries = exp.split_data(
        x,
        sizes["n_train"],
        sizes["n_index"],
        sizes["n_queries"],
        rng,
    )
    distances = exp.distance_matrix(index_x, queries)
    inner_base = np.percentile(distances, 15.0, axis=1)
    outer_base = np.percentile(distances, 25.0, axis=1)
    bandwidth_base = float(np.median(outer_base))
    base_radius = base.median_query_distance_percentile(distances, 35.0)
    query_radius = base.median_query_distance_percentile(
        distances,
        float(COMMON["radius_percentile"]),
    )
    norm_quantile = float(np.percentile(np.linalg.norm(index_x, axis=1), NORM_PERCENTILE))
    delta = max(norm_quantile / DELTA_DIVISIONS, exp.EPS)

    model = exp.train_mech(
        train_x,
        MAX_TABLES,
        MAX_BITS,
        24,
        128,
        1e-3,
        1.0,
        0.1,
        0.1,
        0.1,
        "cpu",
    )
    index = exp.MECHHashIndex(
        model,
        index_x,
        int(COMMON["L"]),
        int(COMMON["K"]),
        int(COMMON["hamming_probe"]),
    )

    rows = []
    total = len(ANNULUS_COUNTS) * len(TOTAL_SAMPLE_BUDGETS)
    done = 0
    for annulus_count in ANNULUS_COUNTS:
        for total_budget in TOTAL_SAMPLE_BUDGETS:
            config = exp.EvalConfig(
                delta=delta,
                radius_factor=1.0,
                bandwidth_factor=float(COMMON["bandwidth_factor"]),
                variant="full",
                query_mode="radius_first",
                min_collisions=int(COMMON["min_collisions"]),
                fixed_query_radius=query_radius,
                annulus_count=int(annulus_count),
                ring_sample_size=max(1, int(total_budget)),
                total_ring_sample_budget=int(total_budget),
                ring_sample_allocation=ALLOCATION_MODE,
            )
            start = time.perf_counter()
            row = exp.evaluate_index(
                index,
                index_x,
                queries,
                distances,
                inner_base,
                outer_base,
                bandwidth_base,
                config,
            )
            row.update(
                {
                    "dataset": dataset,
                    "annulus_count": int(annulus_count),
                    "total_sample_budget": int(total_budget),
                    "allocation_mode": ALLOCATION_MODE,
                    "strict_total_sample_budget": True,
                    "norm_percentile": NORM_PERCENTILE,
                    "delta_divisions": DELTA_DIVISIONS,
                    "norm_quantile": norm_quantile,
                    "delta": delta,
                    "delta_factor_equiv": delta / base_radius,
                    "base_radius": base_radius,
                    "fixed_query_radius": query_radius,
                    "L": COMMON["L"],
                    "K": COMMON["K"],
                    "radius_percentile": COMMON["radius_percentile"],
                    "bandwidth_factor": COMMON["bandwidth_factor"],
                    "hamming_probe": COMMON["hamming_probe"],
                    "min_collisions": COMMON["min_collisions"],
                    "wall_time_s": time.perf_counter() - start,
                }
            )
            rows.append(row)
            done += 1
            print(
                f"[{dataset} {done:02d}/{total}] "
                f"rings={annulus_count} budget={total_budget}: "
                f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                f"F1={row['point_f1']:.3f} KDE={row['kde_abs_relative_error']:.3f} "
                f"actual_sample={row['kde_sample_size']:.1f} "
                f"T={row['query_time_ms']:.3f} KDE_T={row['kde_time_ms']:.3f}",
                flush=True,
            )

    meta = {
        "dataset": dataset,
        "split": sizes,
        "base_radius": base_radius,
        "query_radius": query_radius,
        "norm_percentile": NORM_PERCENTILE,
        "delta_divisions": DELTA_DIVISIONS,
        "norm_quantile": norm_quantile,
        "delta": delta,
        "delta_factor_equiv": delta / base_radius,
        "bandwidth_base": bandwidth_base,
        "allocation_mode": ALLOCATION_MODE,
    }
    return base.score_rows(pd.DataFrame(rows)), meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.OUT_DIR = OUT_DIR
    base.ANNULUS_COUNTS = ANNULUS_COUNTS
    base.TOTAL_SAMPLE_BUDGETS = TOTAL_SAMPLE_BUDGETS
    all_rows = []
    metas = []
    for dataset in DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        df, meta = evaluate_dataset(dataset)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / "annulus_count_weighted_budget.csv", index=False)
        base.plot_dataset_heatmaps(df, dataset)
        all_rows.append(df)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_annulus_count_weighted_budget.csv", index=False)
    base.plot_cross_dataset_metric(
        all_df,
        "kde_abs_relative_error",
        "KDE relative error under strict weighted total-budget sensitivity",
        "kde_error_cross_dataset_heatmaps",
        "viridis",
    )
    base.plot_cross_dataset_metric(
        all_df,
        "query_time_ms",
        "Online time under strict weighted total-budget sensitivity",
        "online_time_cross_dataset_heatmaps",
        "magma",
    )
    base.plot_cross_dataset_metric(
        all_df,
        "kde_time_ms",
        "KDE-stage time under strict weighted total-budget sensitivity",
        "kde_time_cross_dataset_heatmaps",
        "magma",
    )

    best_rows = []
    baseline_rows = []
    for dataset, part in all_df.groupby("dataset"):
        feasible = part[part["feasible"]]
        pool = feasible if not feasible.empty else part
        best = pool.sort_values(
            ["score_kde", "kde_abs_relative_error", "query_time_ms"],
            ascending=[False, True, True],
        ).iloc[0]
        best_rows.append(best)
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == 16)
        ].iloc[0]
        baseline_rows.append(baseline)

    best_df = pd.DataFrame(best_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    best_df.to_csv(OUT_DIR / "best_annulus_count_weighted_budget_by_dataset.csv", index=False)
    baseline_df.to_csv(OUT_DIR / "baseline_annulus_count_weighted_budget_by_dataset.csv", index=False)

    cols = [
        "dataset",
        "annulus_count",
        "total_sample_budget",
        "kde_sample_size",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "signed_kde_bias",
        "candidate_expansion_ratio",
        "query_time_ms",
        "kde_time_ms",
        "score_kde",
    ]
    summary = [
        "# Annulus Count And Strict Weighted Total-Budget Sensitivity",
        "",
        "This experiment fixes the total KDE sampling budget per query and allocates it across rings proportionally to ring size.",
        f"`delta` is fixed as `Q_{NORM_PERCENTILE}(||x||) / {DELTA_DIVISIONS}`.",
        f"`allocation_mode = {ALLOCATION_MODE}`.",
        "",
        "## Baseline Grid Point",
        "",
        base.markdown_table(baseline_df.sort_values("dataset"), cols),
        "",
        "## Best Grid Point By Dataset",
        "",
        base.markdown_table(best_df.sort_values("dataset"), cols),
        "",
        "## Figures",
        "",
        "- `figures/kde_error_cross_dataset_heatmaps.png`",
        "- `figures/online_time_cross_dataset_heatmaps.png`",
        "- `figures/kde_time_cross_dataset_heatmaps.png`",
        "- `figures/<dataset>/annulus_count_total_sample_heatmaps.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "common_parameters": COMMON,
                "max_tables": MAX_TABLES,
                "max_bits": MAX_BITS,
                "norm_percentile": NORM_PERCENTILE,
                "delta_divisions": DELTA_DIVISIONS,
                "annulus_counts": ANNULUS_COUNTS,
                "total_sample_budgets": TOTAL_SAMPLE_BUDGETS,
                "allocation_mode": ALLOCATION_MODE,
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved weighted-budget sensitivity results to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
