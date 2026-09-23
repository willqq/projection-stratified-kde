from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech_annulus_experiments as exp


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("mech_norm_percentile_delta_experiment")

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
    "annulus_count": 1,
    "ring_sample_size": 16,
}

NORM_PERCENTILES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
DIVISIONS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
TARGET_PERCENTILE = 25
TARGET_DIVISIONS = 12

METRICS = [
    ("point_precision", "Precision"),
    ("point_recall", "Recall"),
    ("point_f1", "F1-score"),
    ("kde_abs_relative_error", "KDE relative error"),
    ("candidate_expansion_ratio", "CER"),
    ("query_time_ms", "Online query time (ms)"),
]
COLORS = {
    "isolet": "#2F6B9A",
    "cifar10": "#2F8F83",
    "cifar10_gist512": "#D9A441",
    "amazon": "#B55252",
}


def split_sizes(n_rows: int) -> dict[str, int]:
    if n_rows >= SPLIT["n_train"] + SPLIT["n_index"] + SPLIT["n_queries"]:
        return dict(SPLIT)
    n_queries = min(SPLIT["n_queries"], max(10, n_rows // 20))
    n_train = min(SPLIT["n_train"], max(100, int(n_rows * 0.45)))
    n_index = min(SPLIT["n_index"], n_rows - n_train - n_queries)
    if n_index < 100:
        n_index = max(1, n_rows - n_queries - 100)
        n_train = n_rows - n_queries - n_index
    return {"n_train": int(n_train), "n_index": int(n_index), "n_queries": int(n_queries)}


def median_query_distance_percentile(distances: np.ndarray, percentile: float) -> float:
    return float(np.median(np.percentile(distances, percentile, axis=1)))


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kde_quality"] = (1.0 - df["kde_abs_relative_error"]).clip(lower=0.0, upper=1.0)
    time_max = max(float(df["query_time_ms"].max()), 1e-12)
    df["speed_quality"] = (1.0 - df["query_time_ms"] / time_max).clip(lower=0.0, upper=1.0)
    df["score_highR"] = (
        0.40 * df["point_f1"]
        + 0.35 * df["point_recall"]
        + 0.15 * df["kde_quality"]
        + 0.10 * df["speed_quality"]
    )
    df["feasible"] = (df["point_recall"] >= 0.80) & (df["point_f1"] >= 0.60)
    return df


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, dict]:
    exp.set_seed(exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    x = exp.load_dataset(dataset)
    sizes = split_sizes(len(x))
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
    base_radius = median_query_distance_percentile(distances, 35.0)
    query_radius = median_query_distance_percentile(
        distances,
        float(COMMON["radius_percentile"]),
    )

    index_norms = np.linalg.norm(index_x, axis=1)
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

    settings = []
    for percentile in NORM_PERCENTILES:
        norm_quantile = float(np.percentile(index_norms, percentile))
        delta = max(norm_quantile / TARGET_DIVISIONS, exp.EPS)
        settings.append(
            {
                "scheme": "norm_percentile_fixed12",
                "x_value": float(percentile),
                "norm_percentile": float(percentile),
                "divisions": float(TARGET_DIVISIONS),
                "norm_quantile": norm_quantile,
                "delta": delta,
                "delta_factor_equiv": delta / base_radius,
            }
        )

    norm_quantile_25 = float(np.percentile(index_norms, TARGET_PERCENTILE))
    for divisions in DIVISIONS:
        delta = max(norm_quantile_25 / float(divisions), exp.EPS)
        settings.append(
            {
                "scheme": "norm_p25_variable_divisions",
                "x_value": float(divisions),
                "norm_percentile": float(TARGET_PERCENTILE),
                "divisions": float(divisions),
                "norm_quantile": norm_quantile_25,
                "delta": delta,
                "delta_factor_equiv": delta / base_radius,
            }
        )

    rows = []
    for done, setting in enumerate(settings, start=1):
        config = exp.EvalConfig(
            delta=float(setting["delta"]),
            radius_factor=1.0,
            bandwidth_factor=float(COMMON["bandwidth_factor"]),
            variant="full",
            query_mode="radius_first",
            min_collisions=int(COMMON["min_collisions"]),
            fixed_query_radius=query_radius,
            annulus_count=int(COMMON["annulus_count"]),
            ring_sample_size=int(COMMON["ring_sample_size"]),
        )
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
                **setting,
                "base_radius": base_radius,
                "fixed_query_radius": query_radius,
                "L": COMMON["L"],
                "K": COMMON["K"],
                "radius_percentile": COMMON["radius_percentile"],
                "bandwidth_factor": COMMON["bandwidth_factor"],
                "hamming_probe": COMMON["hamming_probe"],
                "min_collisions": COMMON["min_collisions"],
                "annulus_count": COMMON["annulus_count"],
                "ring_sample_size": COMMON["ring_sample_size"],
            }
        )
        rows.append(row)
        print(
            f"[{dataset} {done:02d}/{len(settings)}] "
            f"{setting['scheme']}={setting['x_value']}: "
            f"delta={setting['delta']:.4f} "
            f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
            f"F1={row['point_f1']:.3f} KDE={row['kde_abs_relative_error']:.3f} "
            f"T={row['query_time_ms']:.3f}",
            flush=True,
        )

    meta = {
        "dataset": dataset,
        "split": sizes,
        "base_radius": base_radius,
        "query_radius": query_radius,
        "norm_p25": norm_quantile_25,
        "target_delta_p25_div12": norm_quantile_25 / TARGET_DIVISIONS,
        "target_delta_factor_equiv": (norm_quantile_25 / TARGET_DIVISIONS) / base_radius,
        "bandwidth_base": bandwidth_base,
    }
    return score_rows(pd.DataFrame(rows)), meta


def plot_scheme(df: pd.DataFrame, scheme: str, x_label: str, out_name: str) -> None:
    part_scheme = df[df["scheme"] == scheme]
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.0))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, METRICS):
        for dataset in DATASETS:
            part = part_scheme[part_scheme["dataset"] == dataset].sort_values("x_value")
            ax.plot(
                part["x_value"].astype(float),
                part[metric].astype(float),
                marker="o",
                markersize=4.0,
                linewidth=1.8,
                label=dataset,
                color=COLORS[dataset],
            )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.28)
    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(out_name.replace("_", " "), x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{out_name}.png", dpi=260)
    fig.savefig(figure_dir / f"{out_name}.pdf")
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
            if isinstance(value, str):
                cells.append(value)
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metas = []
    for dataset in DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        df, meta = evaluate_dataset(dataset)
        all_rows.append(df)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "norm_percentile_delta_results.csv", index=False)

    best_rows = []
    target_rows = []
    for (dataset, scheme), part in all_df.groupby(["dataset", "scheme"]):
        feasible = part[part["feasible"]]
        pool = feasible if not feasible.empty else part
        best = pool.sort_values(
            ["score_highR", "point_f1", "query_time_ms"],
            ascending=[False, False, True],
        ).iloc[0]
        best_rows.append(best)

        if scheme == "norm_percentile_fixed12":
            target = part[part["x_value"] == float(TARGET_PERCENTILE)].iloc[0]
            target_rows.append(target)

    best_df = pd.DataFrame(best_rows)
    target_df = pd.DataFrame(target_rows)
    best_df.to_csv(OUT_DIR / "best_norm_percentile_delta_by_scheme.csv", index=False)
    target_df.to_csv(OUT_DIR / "target_p25_div12_results.csv", index=False)

    plot_scheme(
        all_df,
        "norm_percentile_fixed12",
        "norm percentile p, delta = Q_p(||x||) / 12",
        "norm_percentile_fixed12_delta_comparison",
    )
    plot_scheme(
        all_df,
        "norm_p25_variable_divisions",
        "number of equal divisions, delta = Q_25(||x||) / m",
        "norm_p25_variable_divisions_delta_comparison",
    )

    summary_cols = [
        "dataset",
        "scheme",
        "x_value",
        "norm_percentile",
        "divisions",
        "norm_quantile",
        "delta",
        "delta_factor_equiv",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "candidate_expansion_ratio",
        "query_time_ms",
        "score_highR",
    ]
    summary = [
        "# Norm-Percentile Delta Experiment",
        "",
        "This experiment defines the sphere-layer interval from data-norm percentiles.",
        "",
        f"- Target adaptive setting: `delta = Q_{{{TARGET_PERCENTILE}}}(||x||) / {TARGET_DIVISIONS}`.",
        f"- Percentile sweep: `delta = Q_p(||x||) / {TARGET_DIVISIONS}`.",
        f"- Division sweep: `delta = Q_{{{TARGET_PERCENTILE}}}(||x||) / m`.",
        "",
        "## Target p25-div12 Results",
        "",
        markdown_table(target_df.sort_values("dataset"), summary_cols),
        "",
        "## Best Setting By Dataset And Scheme",
        "",
        markdown_table(best_df.sort_values(["dataset", "scheme"]), summary_cols),
        "",
        "## Figures",
        "",
        "- `figures/norm_percentile_fixed12_delta_comparison.png`",
        "- `figures/norm_p25_variable_divisions_delta_comparison.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "common_parameters": COMMON,
                "max_tables": MAX_TABLES,
                "max_bits": MAX_BITS,
                "norm_percentiles": NORM_PERCENTILES,
                "divisions": DIVISIONS,
                "target_percentile": TARGET_PERCENTILE,
                "target_divisions": TARGET_DIVISIONS,
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved norm-percentile delta experiment to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
