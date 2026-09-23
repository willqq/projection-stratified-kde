from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_annulus_count_total_sample_sensitivity as base
import run_annulus_count_weighted_budget_sensitivity as weighted


OUT_DIR = Path("mech_annulus_count_even_2_20_weighted_budget_sensitivity_2x_anchor")
PREPROCESSING = "minmax scaling followed by 2*x plus the maximum-norm sample"
ANNULUS_COUNTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
TOTAL_SAMPLE_BUDGETS = [2, 4, 8, 16, 32, 64, 128, 256, 512]


def best_by_budget(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in all_df.groupby("dataset"):
        for budget, group in part.groupby("total_sample_budget"):
            best = group.sort_values(["kde_abs_relative_error", "query_time_ms"]).iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "total_sample_budget": int(budget),
                    "best_annulus_count": int(best["annulus_count"]),
                    "best_kde_error": float(best["kde_abs_relative_error"]),
                    "best_query_time_ms": float(best["query_time_ms"]),
                    "best_kde_time_ms": float(best["kde_time_ms"]),
                    "best_score_kde": float(best["score_kde"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    weighted.OUT_DIR = OUT_DIR
    weighted.ANNULUS_COUNTS = ANNULUS_COUNTS
    weighted.TOTAL_SAMPLE_BUDGETS = TOTAL_SAMPLE_BUDGETS
    base.OUT_DIR = OUT_DIR
    base.ANNULUS_COUNTS = ANNULUS_COUNTS
    base.TOTAL_SAMPLE_BUDGETS = TOTAL_SAMPLE_BUDGETS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metas = []
    for dataset in weighted.DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        df, meta = weighted.evaluate_dataset(dataset)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / "annulus_count_even_2_20_weighted_budget.csv", index=False)
        base.plot_dataset_heatmaps(df, dataset)
        all_rows.append(df)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_annulus_count_even_2_20_weighted_budget.csv", index=False)
    base.plot_cross_dataset_metric(
        all_df,
        "kde_abs_relative_error",
        "KDE relative error under annulus count 2-20 and strict weighted total budget",
        "kde_error_cross_dataset_heatmaps",
        "viridis",
    )
    base.plot_cross_dataset_metric(
        all_df,
        "query_time_ms",
        "Online time under annulus count 2-20 and strict weighted total budget",
        "online_time_cross_dataset_heatmaps",
        "magma",
    )
    base.plot_cross_dataset_metric(
        all_df,
        "kde_time_ms",
        "KDE-stage time under annulus count 2-20 and strict weighted total budget",
        "kde_time_cross_dataset_heatmaps",
        "magma",
    )

    feasible = all_df[all_df["feasible"]]
    pool = feasible if not feasible.empty else all_df
    best_df = (
        pool.sort_values(
            ["dataset", "score_kde", "kde_abs_relative_error", "query_time_ms"],
            ascending=[True, False, True, True],
        )
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    by_budget = best_by_budget(all_df)
    best_df.to_csv(OUT_DIR / "best_even_2_20_weighted_budget_by_dataset.csv", index=False)
    by_budget.to_csv(OUT_DIR / "best_even_2_20_annulus_by_budget.csv", index=False)

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
    budget_cols = [
        "dataset",
        "total_sample_budget",
        "best_annulus_count",
        "best_kde_error",
        "best_query_time_ms",
        "best_kde_time_ms",
        "best_score_kde",
    ]
    summary = [
        "# Even Annulus Count 2-20 Strict Weighted Total-Budget Sensitivity",
        "",
        "`annulus_count = [2,4,6,8,10,12,14,16,18,20]`.",
        f"`preprocessing = {PREPROCESSING}`.",
        "The total KDE sampling budget is fixed per query and allocated across annuli proportionally to ring size.",
        f"`delta` is fixed as `Q_{weighted.NORM_PERCENTILE}(||x||) / {weighted.DELTA_DIVISIONS}`.",
        f"`allocation_mode = {weighted.ALLOCATION_MODE}`.",
        "",
        "## Best Grid Point By Dataset",
        "",
        base.markdown_table(best_df.sort_values("dataset"), cols),
        "",
        "## Best Annulus Count By Budget",
        "",
        base.markdown_table(by_budget.sort_values(["dataset", "total_sample_budget"]), budget_cols),
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
                "datasets": weighted.DATASETS,
                "preprocessing": PREPROCESSING,
                "common_parameters": weighted.COMMON,
                "max_tables": weighted.MAX_TABLES,
                "max_bits": weighted.MAX_BITS,
                "norm_percentile": weighted.NORM_PERCENTILE,
                "delta_divisions": weighted.DELTA_DIVISIONS,
                "annulus_counts": ANNULUS_COUNTS,
                "total_sample_budgets": TOTAL_SAMPLE_BUDGETS,
                "allocation_mode": weighted.ALLOCATION_MODE,
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved even annulus-count sensitivity results to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
