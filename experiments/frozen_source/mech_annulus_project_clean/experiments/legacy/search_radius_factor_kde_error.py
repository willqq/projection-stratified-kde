from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech_annulus_experiments as exp
from run_annulus_count_total_sample_sensitivity import split_sizes
from search_query_radius_kde_coverage import gaussian_kernel, load_reference_dataset


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("query_radius_factor_kde_error_search")
BANDWIDTH_FACTOR = 0.25
FACTORS = [round(value, 2) for value in np.arange(0.10, 1.0001, 0.05)]


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exp.set_seed(exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    x = load_reference_dataset(dataset)
    sizes = split_sizes(len(x))
    train_x, index_x, queries = exp.split_data(
        x,
        sizes["n_train"],
        sizes["n_index"],
        sizes["n_queries"],
        rng,
    )
    distances = exp.distance_matrix(index_x, queries)
    bandwidth_base = float(np.median(np.percentile(distances, 25.0, axis=1)))
    bandwidth = bandwidth_base * BANDWIDTH_FACTOR
    weights = gaussian_kernel(distances, bandwidth)
    global_kde = weights.sum(axis=1)
    query_max_radii = np.max(distances, axis=1)
    reference_radius = float(np.median(query_max_radii))

    rows = []
    query_rows = []
    for factor in FACTORS:
        fixed_query_radius = float(reference_radius * factor)
        mask = distances <= fixed_query_radius
        local_kde = (weights * mask).sum(axis=1)
        coverage = local_kde / np.maximum(global_kde, exp.EPS)
        abs_relative_error = np.abs(local_kde - global_kde) / np.maximum(global_kde, exp.EPS)
        signed_relative_error = (local_kde - global_kde) / np.maximum(global_kde, exp.EPS)
        candidate_size = mask.sum(axis=1)
        point_fraction = candidate_size / distances.shape[1]

        rows.append(
            {
                "dataset": dataset,
                "radius_factor": factor,
                "reference_radius_R100": reference_radius,
                "fixed_query_radius": fixed_query_radius,
                "bandwidth_factor": BANDWIDTH_FACTOR,
                "bandwidth": bandwidth,
                "mean_kde_abs_relative_error": float(np.mean(abs_relative_error)),
                "median_kde_abs_relative_error": float(np.median(abs_relative_error)),
                "p90_kde_abs_relative_error": float(np.percentile(abs_relative_error, 90)),
                "max_kde_abs_relative_error": float(np.max(abs_relative_error)),
                "mean_signed_relative_error": float(np.mean(signed_relative_error)),
                "mean_kde_coverage": float(np.mean(coverage)),
                "median_kde_coverage": float(np.median(coverage)),
                "min_kde_coverage": float(np.min(coverage)),
                "mean_global_kde": float(np.mean(global_kde)),
                "mean_local_kde": float(np.mean(local_kde)),
                "mean_candidate_size": float(np.mean(candidate_size)),
                "median_candidate_size": float(np.median(candidate_size)),
                "mean_point_fraction": float(np.mean(point_fraction)),
            }
        )

        for query_id in range(len(queries)):
            query_rows.append(
                {
                    "dataset": dataset,
                    "query_id": query_id,
                    "radius_factor": factor,
                    "reference_radius_R100": reference_radius,
                    "fixed_query_radius": fixed_query_radius,
                    "query_max_radius": float(query_max_radii[query_id]),
                    "global_kde": float(global_kde[query_id]),
                    "local_kde": float(local_kde[query_id]),
                    "kde_abs_relative_error": float(abs_relative_error[query_id]),
                    "signed_relative_error": float(signed_relative_error[query_id]),
                    "kde_coverage": float(coverage[query_id]),
                    "candidate_size": int(candidate_size[query_id]),
                    "point_fraction": float(point_fraction[query_id]),
                }
            )

    meta = {
        "dataset": dataset,
        "preprocessing": "reference_minmax_without_origin_shift",
        "split": sizes,
        "n_index": int(len(index_x)),
        "n_queries": int(len(queries)),
        "bandwidth_base": bandwidth_base,
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "bandwidth": bandwidth,
        "reference_radius_R100": reference_radius,
        "factor_definition": "fixed_query_radius = radius_factor * median(max_query_to_index_distance)",
    }
    return pd.DataFrame(rows), pd.DataFrame(query_rows), meta


def pivot_error_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(
        index="radius_factor",
        columns="dataset",
        values="mean_kde_abs_relative_error",
        aggfunc="mean",
    ).reindex(index=FACTORS, columns=DATASETS)
    table.to_csv(OUT_DIR / "radius_factor_kde_error_table.csv")
    return table


def plot_error_table(table: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    display = table.applymap(lambda value: f"{float(value):.4f}")

    fig, ax = plt.subplots(figsize=(8.8, 10.2))
    ax.axis("off")
    table_artist = ax.table(
        cellText=display.reset_index().values,
        colLabels=["factor"] + list(display.columns),
        cellLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8.5)
    table_artist.scale(1.0, 1.24)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#fafafa")
            cell.set_text_props(weight="bold")
    ax.set_title(
        "Mean KDE relative error by radius factor",
        fontsize=12,
        fontweight="bold",
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "radius_factor_kde_error_table.png", dpi=240)
    fig.savefig(figure_dir / "radius_factor_kde_error_table.pdf")
    plt.close(fig)


def plot_error_curves(df: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "isolet": "#4c78a8",
        "cifar10": "#59a14f",
        "cifar10_gist512": "#f28e2b",
        "amazon": "#e15759",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    for dataset, part in df.groupby("dataset"):
        part = part.sort_values("radius_factor")
        axes[0].plot(
            part["radius_factor"],
            part["mean_kde_abs_relative_error"],
            marker="o",
            linewidth=1.8,
            label=dataset,
            color=colors.get(dataset),
        )
        axes[1].plot(
            part["radius_factor"],
            part["mean_candidate_size"],
            marker="o",
            linewidth=1.8,
            label=dataset,
            color=colors.get(dataset),
        )
    axes[0].set_title("KDE estimation error")
    axes[0].set_ylabel("mean abs. relative error")
    axes[1].set_title("Mean candidate size")
    axes[1].set_ylabel("points")
    for ax in axes:
        ax.set_xlabel("radius factor")
        ax.set_xticks(FACTORS[::2])
        ax.grid(True, alpha=0.28)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Radius factor search for KDE approximation",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "radius_factor_kde_error_curves.png", dpi=240)
    fig.savefig(figure_dir / "radius_factor_kde_error_curves.pdf")
    plt.close(fig)


def format_markdown_table(table: pd.DataFrame) -> str:
    display = table.reset_index().copy()
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col == "radius_factor":
                cells.append(f"{float(value):.2f}")
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        part = part.sort_values("radius_factor")
        for threshold in [0.10, 0.05, 0.02, 0.01]:
            candidates = part[part["mean_kde_abs_relative_error"] <= threshold]
            if candidates.empty:
                selected = part.iloc[-1]
                met = False
            else:
                selected = candidates.iloc[0]
                met = True
            rows.append(
                {
                    "dataset": dataset,
                    "error_threshold": threshold,
                    "target_met": bool(met),
                    "recommended_factor": float(selected["radius_factor"]),
                    "fixed_query_radius": float(selected["fixed_query_radius"]),
                    "mean_kde_abs_relative_error": float(selected["mean_kde_abs_relative_error"]),
                    "mean_kde_coverage": float(selected["mean_kde_coverage"]),
                    "mean_candidate_size": float(selected["mean_candidate_size"]),
                }
            )
    return pd.DataFrame(rows)


def write_summary(table: pd.DataFrame, all_df: pd.DataFrame, thresholds: pd.DataFrame, metas: list[dict]) -> None:
    threshold_show = thresholds[thresholds["error_threshold"].isin([0.05, 0.02])].copy()
    lines = [
        "# Radius Factor KDE Error Search",
        "",
        "本实验按数据集单独设置半径，使用旧实验一致的 MinMax 预处理和 train/index/query 划分。半径因子定义为：",
        "",
        "```text",
        "R(factor) = factor * R100",
        "R100 = median_q max_x ||x - q||",
        "```",
        "",
        f"`factor` 从 0.10 到 1.00，步长 0.05。核带宽沿用公共实验的 `bandwidth_factor={BANDWIDTH_FACTOR}`。误差定义为 `mean(|KDE_R-KDE_global|/KDE_global)`。",
        "",
        "## Mean KDE Error Table",
        "",
        format_markdown_table(table),
        "",
        "## Threshold Recommendations",
        "",
        "| dataset | error_threshold | target_met | recommended_factor | fixed_query_radius | mean_kde_abs_relative_error | mean_kde_coverage | mean_candidate_size |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in threshold_show.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    f"{float(row['error_threshold']):.2f}",
                    str(bool(row["target_met"])),
                    f"{float(row['recommended_factor']):.2f}",
                    f"{float(row['fixed_query_radius']):.4f}",
                    f"{float(row['mean_kde_abs_relative_error']):.4f}",
                    f"{float(row['mean_kde_coverage']):.4f}",
                    f"{float(row['mean_candidate_size']):.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `radius_factor_kde_error_all.csv`",
            "- `radius_factor_kde_error_by_query.csv`",
            "- `radius_factor_kde_error_table.csv`",
            "- `radius_factor_threshold_recommendations.csv`",
            "- `figures/radius_factor_kde_error_table.png`",
            "- `figures/radius_factor_kde_error_curves.png`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "experiment_config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "radius_factors": FACTORS,
                "bandwidth_factor": BANDWIDTH_FACTOR,
                "factor_definition": "fixed_query_radius = factor * median(max_query_to_index_distance)",
                "error_metric": "mean(|KDE_R - KDE_global| / KDE_global)",
                "dataset_meta": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_query_rows = []
    metas = []
    for dataset in DATASETS:
        print(f"=== Dataset: {dataset} ===", flush=True)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df, query_df, meta = evaluate_dataset(dataset)
        df.to_csv(dataset_dir / "radius_factor_kde_error.csv", index=False)
        query_df.to_csv(dataset_dir / "radius_factor_kde_error_by_query.csv", index=False)
        all_rows.append(df)
        all_query_rows.append(query_df)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_query_df = pd.concat(all_query_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "radius_factor_kde_error_all.csv", index=False)
    all_query_df.to_csv(OUT_DIR / "radius_factor_kde_error_by_query.csv", index=False)
    table = pivot_error_table(all_df)
    thresholds = summarize_thresholds(all_df)
    thresholds.to_csv(OUT_DIR / "radius_factor_threshold_recommendations.csv", index=False)
    plot_error_table(table)
    plot_error_curves(all_df)
    write_summary(table, all_df, thresholds, metas)
    print(f"Saved radius factor KDE error search to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
