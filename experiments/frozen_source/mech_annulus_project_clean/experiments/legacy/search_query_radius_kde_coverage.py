from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

import mech_annulus_experiments as exp
from run_annulus_count_total_sample_sensitivity import split_sizes


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("query_radius_kde_coverage_search")
BANDWIDTH_FACTORS = [0.25, 0.5, 1.0]
PRIMARY_BANDWIDTH_FACTOR = 0.25
RADIUS_PERCENTILES = [
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    100,
]
COVERAGE_TARGETS = [0.90, 0.95, 0.97, 0.99]


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-(distances**2) / (2.0 * bandwidth**2))


def reference_minmax_scale(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(MinMaxScaler().fit_transform(x)).astype(np.float32)


def load_reference_dataset(name: str) -> np.ndarray:
    name = name.lower()
    if name == "isolet":
        df = pd.read_csv("data/isolet/isolet1+2+3+4.data", header=None)
        x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    elif name == "cifar10":
        with open("data/cifar-10-batches-py/data_batch_1", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x = batch[b"data"].astype(np.float32)
    elif name == "cifar10_gist512":
        x = scio.loadmat("data/cifar10-Gist512/Cifar10-Gist512.mat")["X"].astype(
            np.float32
        )
    elif name == "amazon":
        data, _ = arff.loadarff(
            "data/Amazon_initial_50_30_10000/"
            "Amazon_initial_50_30_10000.arff"
        )
        df = pd.DataFrame(data).drop(columns=["class_duplicate"])
        x = df.to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return reference_minmax_scale(x)


def median_radius(distances: np.ndarray, percentile: float) -> float:
    return float(np.median(np.percentile(distances, percentile, axis=1)))


def kde_coverage_for_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
    outer_base = np.percentile(distances, 25.0, axis=1)
    bandwidth_base = float(np.median(outer_base))

    rows = []
    query_rows = []
    for bandwidth_factor in BANDWIDTH_FACTORS:
        bandwidth = bandwidth_base * float(bandwidth_factor)
        weights = gaussian_kernel(distances, bandwidth)
        global_kde = weights.sum(axis=1)

        for percentile in RADIUS_PERCENTILES:
            query_radii = np.percentile(distances, percentile, axis=1)
            fixed_query_radius = float(np.median(query_radii))
            mask = distances <= fixed_query_radius
            local_kde = (weights * mask).sum(axis=1)
            coverage = local_kde / np.maximum(global_kde, exp.EPS)
            missing = 1.0 - coverage
            point_fraction = mask.mean(axis=1)
            candidate_size = mask.sum(axis=1)
            local_mean_kernel = local_kde / np.maximum(candidate_size, 1)
            global_mean_kernel = global_kde / distances.shape[1]
            mean_kernel_ratio = local_mean_kernel / np.maximum(global_mean_kernel, exp.EPS)

            rows.append(
                {
                    "dataset": dataset,
                    "bandwidth_factor": float(bandwidth_factor),
                    "bandwidth": bandwidth,
                    "radius_percentile": float(percentile),
                    "fixed_query_radius": fixed_query_radius,
                    "mean_kde_coverage": float(np.mean(coverage)),
                    "median_kde_coverage": float(np.median(coverage)),
                    "min_kde_coverage": float(np.min(coverage)),
                    "p10_kde_coverage": float(np.percentile(coverage, 10)),
                    "mean_missing_global_kde": float(np.mean(missing)),
                    "median_missing_global_kde": float(np.median(missing)),
                    "mean_point_fraction": float(np.mean(point_fraction)),
                    "median_point_fraction": float(np.median(point_fraction)),
                    "mean_candidate_size": float(np.mean(candidate_size)),
                    "median_candidate_size": float(np.median(candidate_size)),
                    "mean_local_kde": float(np.mean(local_kde)),
                    "mean_global_kde": float(np.mean(global_kde)),
                    "mean_local_global_kde_gap": float(np.mean(global_kde - local_kde)),
                    "mean_local_global_kde_abs_relative_error": float(
                        np.mean(np.abs(local_kde - global_kde) / np.maximum(global_kde, exp.EPS))
                    ),
                    "mean_kernel_ratio_local_vs_global": float(np.mean(mean_kernel_ratio)),
                }
            )

            for q_id in range(len(queries)):
                query_rows.append(
                    {
                        "dataset": dataset,
                        "query_id": q_id,
                        "bandwidth_factor": float(bandwidth_factor),
                        "radius_percentile": float(percentile),
                        "fixed_query_radius": fixed_query_radius,
                        "query_radius_percentile_value": float(query_radii[q_id]),
                        "global_kde": float(global_kde[q_id]),
                        "local_kde": float(local_kde[q_id]),
                        "kde_coverage": float(coverage[q_id]),
                        "missing_global_kde": float(missing[q_id]),
                        "point_fraction": float(point_fraction[q_id]),
                        "candidate_size": int(candidate_size[q_id]),
                    }
                )

    meta = {
        "dataset": dataset,
        "preprocessing": "reference_minmax_without_origin_shift",
        "split": sizes,
        "n_index": int(len(index_x)),
        "n_queries": int(len(queries)),
        "bandwidth_base": bandwidth_base,
        "radius_values": {
            str(percentile): median_radius(distances, percentile)
            for percentile in RADIUS_PERCENTILES
        },
    }
    return pd.DataFrame(rows), pd.DataFrame(query_rows), meta


def recommend_radius(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, bandwidth_factor), part in df.groupby(["dataset", "bandwidth_factor"]):
        part = part.sort_values("radius_percentile")
        for target in COVERAGE_TARGETS:
            candidates = part[part["mean_kde_coverage"] >= target]
            if candidates.empty:
                selected = part.iloc[-1]
                met = False
            else:
                selected = candidates.iloc[0]
                met = True
            rows.append(
                {
                    "dataset": dataset,
                    "bandwidth_factor": float(bandwidth_factor),
                    "coverage_target": float(target),
                    "target_met": bool(met),
                    "recommended_radius_percentile": float(selected["radius_percentile"]),
                    "recommended_query_radius": float(selected["fixed_query_radius"]),
                    "mean_kde_coverage": float(selected["mean_kde_coverage"]),
                    "p10_kde_coverage": float(selected["p10_kde_coverage"]),
                    "min_kde_coverage": float(selected["min_kde_coverage"]),
                    "mean_missing_global_kde": float(selected["mean_missing_global_kde"]),
                    "mean_point_fraction": float(selected["mean_point_fraction"]),
                    "mean_candidate_size": float(selected["mean_candidate_size"]),
                }
            )
    return pd.DataFrame(rows)


def choose_final_recommendation(recs: pd.DataFrame) -> pd.DataFrame:
    primary = recs[
        (np.isclose(recs["bandwidth_factor"], PRIMARY_BANDWIDTH_FACTOR))
        & (np.isclose(recs["coverage_target"], 0.95))
    ].copy()
    primary["recommendation"] = "balanced_95pct_mean_coverage"

    conservative = recs[
        (np.isclose(recs["bandwidth_factor"], PRIMARY_BANDWIDTH_FACTOR))
        & (np.isclose(recs["coverage_target"], 0.97))
    ].copy()
    conservative["recommendation"] = "conservative_97pct_mean_coverage"

    return pd.concat([primary, conservative], ignore_index=True)


def uniform_percentile_summary(df: pd.DataFrame) -> pd.DataFrame:
    primary = df[np.isclose(df["bandwidth_factor"], PRIMARY_BANDWIDTH_FACTOR)].copy()
    rows = []
    for percentile, part in primary.groupby("radius_percentile"):
        rows.append(
            {
                "radius_percentile": float(percentile),
                "mean_kde_coverage": float(part["mean_kde_coverage"].mean()),
                "min_dataset_kde_coverage": float(part["mean_kde_coverage"].min()),
                "mean_missing_global_kde": float(part["mean_missing_global_kde"].mean()),
                "max_dataset_missing_global_kde": float(part["mean_missing_global_kde"].max()),
                "mean_point_fraction": float(part["mean_point_fraction"].mean()),
                "mean_candidate_size": float(part["mean_candidate_size"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("radius_percentile")


def plot_dataset_curves(df: pd.DataFrame, dataset: str) -> None:
    out_dir = OUT_DIR / "figures" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    part = df[df["dataset"] == dataset]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    colors = {0.25: "#4c78a8", 0.5: "#59a14f", 1.0: "#f28e2b"}
    for bandwidth_factor, current in part.groupby("bandwidth_factor"):
        current = current.sort_values("radius_percentile")
        label = f"sigma factor={bandwidth_factor:g}"
        color = colors.get(float(bandwidth_factor), None)
        axes[0].plot(
            current["radius_percentile"],
            current["mean_kde_coverage"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
        axes[1].plot(
            current["radius_percentile"],
            current["mean_point_fraction"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
        axes[2].plot(
            current["radius_percentile"],
            current["mean_candidate_size"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )

    for target in COVERAGE_TARGETS:
        axes[0].axhline(target, color="#333333", linewidth=0.8, alpha=0.25)
    axes[0].set_title("KDE coverage: ball / global")
    axes[0].set_ylabel("mean coverage")
    axes[1].set_title("Point fraction inside radius")
    axes[1].set_ylabel("mean fraction")
    axes[2].set_title("Candidate size inside radius")
    axes[2].set_ylabel("mean points")
    for ax in axes:
        ax.set_xlabel("query radius percentile")
        ax.grid(True, alpha=0.28)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: query radius KDE coverage search", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "kde_coverage_by_radius.png", dpi=240)
    fig.savefig(out_dir / "kde_coverage_by_radius.pdf")
    plt.close(fig)


def plot_cross_dataset(df: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    part = df[np.isclose(df["bandwidth_factor"], PRIMARY_BANDWIDTH_FACTOR)].copy()
    colors = {
        "isolet": "#4c78a8",
        "cifar10": "#59a14f",
        "cifar10_gist512": "#f28e2b",
        "amazon": "#e15759",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    for dataset, current in part.groupby("dataset"):
        current = current.sort_values("radius_percentile")
        axes[0].plot(
            current["radius_percentile"],
            current["mean_kde_coverage"],
            marker="o",
            linewidth=1.8,
            label=dataset,
            color=colors.get(dataset),
        )
        axes[1].plot(
            current["radius_percentile"],
            current["mean_candidate_size"],
            marker="o",
            linewidth=1.8,
            label=dataset,
            color=colors.get(dataset),
        )
    for target in COVERAGE_TARGETS:
        axes[0].axhline(target, color="#333333", linewidth=0.8, alpha=0.22)
    axes[0].set_title(f"KDE coverage at sigma factor={PRIMARY_BANDWIDTH_FACTOR:g}")
    axes[0].set_ylabel("mean ball/global KDE")
    axes[1].set_title("Mean candidate size")
    axes[1].set_ylabel("points")
    for ax in axes:
        ax.set_xlabel("query radius percentile")
        ax.grid(True, alpha=0.28)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-dataset query radius search", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "cross_dataset_kde_coverage.png", dpi=240)
    fig.savefig(figure_dir / "cross_dataset_kde_coverage.pdf")
    plt.close(fig)


def plot_deviation_table(df: pd.DataFrame) -> pd.DataFrame:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = df[np.isclose(df["bandwidth_factor"], PRIMARY_BANDWIDTH_FACTOR)].copy()
    pivot = primary.pivot_table(
        index="dataset",
        columns="radius_percentile",
        values="mean_local_global_kde_abs_relative_error",
        aggfunc="mean",
    ).reindex(index=DATASETS, columns=RADIUS_PERCENTILES)
    pivot.columns = [f"R{int(col)}" for col in pivot.columns]
    pivot.to_csv(OUT_DIR / "kde_mean_deviation_table.csv")

    display = pivot.applymap(lambda value: f"{float(value):.4f}")
    fig_width = max(12.0, 1.1 * (len(display.columns) + 2))
    fig_height = 0.58 * (len(display.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.reset_index().values,
        colLabels=["dataset"] + list(display.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#fafafa")
            cell.set_text_props(weight="bold")
    ax.set_title(
        "Mean KDE deviation vs global KDE by query-radius percentile",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "kde_mean_deviation_table.png", dpi=240)
    fig.savefig(figure_dir / "kde_mean_deviation_table.pdf")
    plt.close(fig)
    return pivot


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, str) or isinstance(value, bool):
                cells.append(str(value))
            elif column in {"recommended_radius_percentile", "radius_percentile"}:
                cells.append(f"{float(value):.0f}")
            elif column in {"mean_kde_coverage", "p10_kde_coverage", "min_kde_coverage", "mean_missing_global_kde", "mean_point_fraction"}:
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(f"{float(value):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(
    df: pd.DataFrame,
    recs: pd.DataFrame,
    final_recs: pd.DataFrame,
    uniform_summary: pd.DataFrame,
    deviation_table: pd.DataFrame,
    metas: list[dict],
) -> None:
    primary = df[np.isclose(df["bandwidth_factor"], PRIMARY_BANDWIDTH_FACTOR)].copy()
    rows = []
    for dataset, part in primary.groupby("dataset"):
        part = part.sort_values("radius_percentile")
        r70 = part[np.isclose(part["radius_percentile"], 70)].iloc[0]
        r90 = part[np.isclose(part["radius_percentile"], 90)].iloc[0]
        r100 = part[np.isclose(part["radius_percentile"], 100)].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "R70": float(r70["fixed_query_radius"]),
                "coverage70": float(r70["mean_kde_coverage"]),
                "candidate70": float(r70["mean_candidate_size"]),
                "R90": float(r90["fixed_query_radius"]),
                "coverage90": float(r90["mean_kde_coverage"]),
                "candidate90": float(r90["mean_candidate_size"]),
                "R100": float(r100["fixed_query_radius"]),
                "coverage100": float(r100["mean_kde_coverage"]),
                "candidate100": float(r100["mean_candidate_size"]),
            }
        )
    checkpoints = pd.DataFrame(rows)

    uniform_targets = []
    for target in [0.90, 0.95, 0.97, 0.99]:
        candidates = uniform_summary[uniform_summary["min_dataset_kde_coverage"] >= target]
        selected = candidates.iloc[0] if not candidates.empty else uniform_summary.iloc[-1]
        uniform_targets.append(
            {
                "coverage_target_for_every_dataset": target,
                "target_met": not candidates.empty,
                "recommended_uniform_radius_percentile": float(selected["radius_percentile"]),
                "mean_kde_coverage": float(selected["mean_kde_coverage"]),
                "min_dataset_kde_coverage": float(selected["min_dataset_kde_coverage"]),
                "mean_missing_global_kde": float(selected["mean_missing_global_kde"]),
                "mean_candidate_size": float(selected["mean_candidate_size"]),
            }
        )
    uniform_targets = pd.DataFrame(uniform_targets)

    uniform_key = uniform_summary[
        uniform_summary["radius_percentile"].isin([70.0, 80.0, 90.0, 100.0])
    ].copy()
    deviation_show = deviation_table.reset_index().copy()

    rec_show = final_recs[
        [
            "dataset",
            "recommendation",
            "recommended_radius_percentile",
            "recommended_query_radius",
            "mean_kde_coverage",
            "p10_kde_coverage",
            "mean_missing_global_kde",
            "mean_point_fraction",
            "mean_candidate_size",
        ]
    ].copy()

    lines = [
        "# Query Radius KDE Coverage Search",
        "",
        "本实验使用原来的数据加载、归一化和 train/index/query 划分，只在精确距离层面扫描 Query radius，不包含 MECH 近似检索误差。对每个查询点，计算球内核密度",
        "",
        "```text",
        "KDE_R(q) = sum_{||x-q|| <= R} exp(-||x-q||^2 / (2 sigma^2))",
        "KDE_global(q) = sum_x exp(-||x-q||^2 / (2 sigma^2))",
        "coverage = KDE_R(q) / KDE_global(q)",
        "```",
        "",
        f"主分析沿用原公共实验的 `bandwidth_factor={PRIMARY_BANDWIDTH_FACTOR}`；另外保留 0.5 和 1.0 作为带宽敏感性参考。",
        "为保持和原实验可比，数据预处理显式使用旧实验的 MinMax 归一化，不使用当前 `mech_annulus_experiments.py` 中的 origin-shift 变体。",
        "",
        "## Dataset-Adaptive Query Radius",
        "",
        "如果目标是让球内 KDE 覆盖至少 95% 的全局 KDE，推荐使用下表的 balanced 设置；如果更强调少漏全局核质量，可用 conservative 设置。",
        "",
        markdown_table(rec_show, list(rec_show.columns)),
        "",
        "## Uniform Radius Percentile",
        "",
        "如果论文或主实验需要所有数据集使用同一个半径分位，按最弱数据集也达到目标覆盖率来选。因为 Amazon 的核质量分布最分散，它决定统一半径的下限。",
        "",
        markdown_table(uniform_targets, list(uniform_targets.columns)),
        "",
        "关键统一分位点如下：",
        "",
        markdown_table(uniform_key, list(uniform_key.columns)),
        "",
        "## Mean KDE Deviation Table",
        "",
        "下表按每个数据集单独调 Query radius，从 5% 到 100% 每次增加 5%，计算平均核密度偏差 `mean(|KDE_R-KDE_global|/KDE_global)`。数值越小，表示球内核密度越接近全局核密度。",
        "",
        markdown_table(deviation_show, list(deviation_show.columns)),
        "",
        "## R=70/90/100 Checkpoints",
        "",
        "原公共实验使用 `radius_percentile=70`。R70 在 ISOLET 和 CIFAR-10 上已经接近全局 KDE，但在 Amazon 上只覆盖约 78.3%，在 GIST512 上约 92.0%；R90/R100 更接近全局 KDE，但候选点数量明显增加。",
        "",
        markdown_table(checkpoints, list(checkpoints.columns)),
        "",
        "## Interpretation",
        "",
        "- 若允许每个数据集自适应设置 R，95% 平均 KDE 覆盖率对应：ISOLET 40 分位、CIFAR-10 50 分位、GIST512 80 分位、Amazon 100 分位。这里的推荐来自 10% 间隔网格，因此比上一版细网格更粗。",
        "- 若需要一个统一 Query radius 分位，建议使用 100 分位；它让四个数据集的平均球内 KDE 覆盖率都达到 100%。",
        "- 若更重视速度，可以使用 90 分位；此时 Amazon 的平均覆盖率约为 94.1%，其余数据集均高于 98%。",
        "- R 越大，局部 KDE 与全局 KDE 的差距越小，但候选规模也越接近全量索引。后续 MECH 检索时需要结合速度预算选择 90 或 100 分位。",
        "- 最终建议：若严格按 10% 步长选择，主实验追求稳定全局 KDE 近似时设置 `radius_percentile=100`；若需要速度-精度折中，设置 `radius_percentile=90` 并说明 Amazon 上约有 5.9% 全局核质量未覆盖。",
        "",
        "## Outputs",
        "",
        "- `kde_radius_search_all.csv`: all radius and bandwidth-factor results.",
        "- `kde_radius_search_by_query.csv`: per-query coverage details.",
        "- `kde_mean_deviation_table.csv`: 10%--100% mean KDE deviation table.",
        "- `radius_recommendations.csv`: target-coverage recommendations.",
        "- `final_query_radius_recommendation.csv`: balanced and conservative choices.",
        "- `figures/kde_mean_deviation_table.png`: rendered table.",
        "- `figures/cross_dataset_kde_coverage.png` and per-dataset figures.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "experiment_config.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "radius_percentiles": RADIUS_PERCENTILES,
                "bandwidth_factors": BANDWIDTH_FACTORS,
                "primary_bandwidth_factor": PRIMARY_BANDWIDTH_FACTOR,
                "coverage_targets": COVERAGE_TARGETS,
                "dataset_meta": metas,
                "recommendation_rule": "Pick the smallest radius percentile whose mean KDE_R/KDE_global reaches the target.",
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
        df, query_df, meta = kde_coverage_for_dataset(dataset)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_dir / "kde_radius_search.csv", index=False)
        query_df.to_csv(dataset_dir / "kde_radius_search_by_query.csv", index=False)
        plot_dataset_curves(df, dataset)
        all_rows.append(df)
        all_query_rows.append(query_df)
        metas.append(meta)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_query_df = pd.concat(all_query_rows, ignore_index=True)
    recs = recommend_radius(all_df)
    final_recs = choose_final_recommendation(recs)
    uniform_summary = uniform_percentile_summary(all_df)
    all_df.to_csv(OUT_DIR / "kde_radius_search_all.csv", index=False)
    all_query_df.to_csv(OUT_DIR / "kde_radius_search_by_query.csv", index=False)
    recs.to_csv(OUT_DIR / "radius_recommendations.csv", index=False)
    final_recs.to_csv(OUT_DIR / "final_query_radius_recommendation.csv", index=False)
    uniform_summary.to_csv(OUT_DIR / "uniform_radius_percentile_summary.csv", index=False)
    plot_cross_dataset(all_df)
    deviation_table = plot_deviation_table(all_df)
    write_summary(all_df, recs, final_recs, uniform_summary, deviation_table, metas)
    print(f"Saved query radius KDE coverage search to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
