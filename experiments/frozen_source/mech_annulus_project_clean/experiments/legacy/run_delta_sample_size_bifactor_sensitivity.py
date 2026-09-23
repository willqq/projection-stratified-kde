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
OUT_DIR = Path("mech_delta_sample_size_bifactor_sensitivity")

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
}

NORM_PERCENTILE = 25
DIVISIONS = [4, 6, 8, 10, 12, 16, 20, 24, 30, 36]
SAMPLE_SIZES = [2, 4, 8, 16, 32, 64, 128]
BASELINE_DIVISIONS = 12
BASELINE_SAMPLE_SIZE = 16


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
    df["score_kde"] = (
        0.55 * df["kde_quality"]
        + 0.20 * df["point_recall"]
        + 0.15 * df["point_f1"]
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
    norm_quantile = float(np.percentile(np.linalg.norm(index_x, axis=1), NORM_PERCENTILE))

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
    total = len(DIVISIONS) * len(SAMPLE_SIZES)
    done = 0
    for divisions in DIVISIONS:
        delta = max(norm_quantile / float(divisions), exp.EPS)
        for sample_size in SAMPLE_SIZES:
            config = exp.EvalConfig(
                delta=delta,
                radius_factor=1.0,
                bandwidth_factor=float(COMMON["bandwidth_factor"]),
                variant="full",
                query_mode="radius_first",
                min_collisions=int(COMMON["min_collisions"]),
                fixed_query_radius=query_radius,
                annulus_count=int(COMMON["annulus_count"]),
                ring_sample_size=int(sample_size),
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
                    "norm_percentile": NORM_PERCENTILE,
                    "divisions": int(divisions),
                    "ring_sample_size": int(sample_size),
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
                    "annulus_count": COMMON["annulus_count"],
                }
            )
            rows.append(row)
            done += 1
            print(
                f"[{dataset} {done:02d}/{total}] "
                f"m={divisions} sample={sample_size} delta={delta:.4f}: "
                f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                f"F1={row['point_f1']:.3f} KDE={row['kde_abs_relative_error']:.3f} "
                f"sampled={row['kde_sample_size']:.1f} T={row['query_time_ms']:.3f}",
                flush=True,
            )

    meta = {
        "dataset": dataset,
        "split": sizes,
        "base_radius": base_radius,
        "query_radius": query_radius,
        "norm_percentile": NORM_PERCENTILE,
        "norm_quantile": norm_quantile,
        "baseline_delta": norm_quantile / BASELINE_DIVISIONS,
        "baseline_delta_factor_equiv": (norm_quantile / BASELINE_DIVISIONS) / base_radius,
        "bandwidth_base": bandwidth_base,
    }
    return score_rows(pd.DataFrame(rows)), meta


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.pivot_table(
            index="ring_sample_size",
            columns="divisions",
            values=metric,
            aggfunc="mean",
        )
        .reindex(index=SAMPLE_SIZES, columns=DIVISIONS)
    )


def add_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: str,
    baseline_marker: bool = True,
) -> None:
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(DIVISIONS)))
    ax.set_xticklabels(DIVISIONS)
    ax.set_yticks(np.arange(len(SAMPLE_SIZES)))
    ax.set_yticklabels(SAMPLE_SIZES)
    ax.set_xlabel("divisions m, delta = Q25(||x||) / m")
    ax.set_ylabel("ring sample size")
    if baseline_marker and BASELINE_DIVISIONS in DIVISIONS and BASELINE_SAMPLE_SIZE in SAMPLE_SIZES:
        x = DIVISIONS.index(BASELINE_DIVISIONS)
        y = SAMPLE_SIZES.index(BASELINE_SAMPLE_SIZE)
        ax.scatter([x], [y], marker="x", s=80, c="black", linewidths=2.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_dataset_heatmaps(df: pd.DataFrame, dataset: str) -> None:
    figure_dir = OUT_DIR / "figures" / dataset
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("kde_abs_relative_error", "KDE relative error", "viridis"),
        ("signed_kde_bias", "Signed KDE bias", "coolwarm"),
        ("query_time_ms", "Online query time (ms)", "magma"),
        ("kde_sample_size", "Actual KDE sample size", "cividis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    axes = axes.ravel()
    for ax, (metric, title, cmap) in zip(axes, metrics):
        add_heatmap(ax, pivot_metric(df, metric), title, cmap)
    fig.suptitle(
        f"{dataset}: delta and KDE sample-size bifactor sensitivity",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "delta_sample_bifactor_heatmaps.png", dpi=260)
    fig.savefig(figure_dir / "delta_sample_bifactor_heatmaps.pdf")
    plt.close(fig)


def plot_cross_dataset_metric(all_df: pd.DataFrame, metric: str, title: str, out_name: str, cmap: str) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        part = all_df[all_df["dataset"] == dataset]
        add_heatmap(ax, pivot_metric(part, metric), dataset, cmap)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
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
            elif column in {"divisions", "ring_sample_size", "L", "K"}:
                cells.append(str(int(value)))
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
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / "delta_sample_bifactor.csv", index=False)
        plot_dataset_heatmaps(df, dataset)
        all_rows.append(df)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_delta_sample_bifactor.csv", index=False)
    plot_cross_dataset_metric(
        all_df,
        "kde_abs_relative_error",
        "KDE relative error under delta and sample-size bifactor sensitivity",
        "kde_error_cross_dataset_heatmaps",
        "viridis",
    )
    plot_cross_dataset_metric(
        all_df,
        "query_time_ms",
        "Online time under delta and sample-size bifactor sensitivity",
        "online_time_cross_dataset_heatmaps",
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
            (part["divisions"] == BASELINE_DIVISIONS)
            & (part["ring_sample_size"] == BASELINE_SAMPLE_SIZE)
        ].iloc[0]
        baseline_rows.append(baseline)

    best_df = pd.DataFrame(best_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    best_df.to_csv(OUT_DIR / "best_delta_sample_by_dataset.csv", index=False)
    baseline_df.to_csv(OUT_DIR / "baseline_delta_sample_by_dataset.csv", index=False)

    cols = [
        "dataset",
        "divisions",
        "ring_sample_size",
        "delta",
        "delta_factor_equiv",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "signed_kde_bias",
        "candidate_expansion_ratio",
        "kde_sample_size",
        "query_time_ms",
        "score_kde",
    ]
    summary = [
        "# Delta And KDE Sample-Size Bifactor Sensitivity",
        "",
        f"`delta` is defined adaptively as `Q_{NORM_PERCENTILE}(||x||) / m`, where `m` is the number of equal divisions.",
        f"The baseline setting is `m={BASELINE_DIVISIONS}` and `ring_sample_size={BASELINE_SAMPLE_SIZE}`.",
        "",
        "## Baseline Grid Point",
        "",
        markdown_table(baseline_df.sort_values("dataset"), cols),
        "",
        "## Best Grid Point By Dataset",
        "",
        markdown_table(best_df.sort_values("dataset"), cols),
        "",
        "## Figures",
        "",
        "- `figures/kde_error_cross_dataset_heatmaps.png`",
        "- `figures/online_time_cross_dataset_heatmaps.png`",
        "- `figures/<dataset>/delta_sample_bifactor_heatmaps.png`",
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
                "divisions": DIVISIONS,
                "sample_sizes": SAMPLE_SIZES,
                "baseline_divisions": BASELINE_DIVISIONS,
                "baseline_sample_size": BASELINE_SAMPLE_SIZE,
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved bifactor sensitivity results to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
