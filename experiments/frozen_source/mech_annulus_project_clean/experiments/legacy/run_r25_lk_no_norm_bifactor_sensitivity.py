from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_r25_l20_k2_128_step2_precision_recall_ratio as no_norm
import run_r25_lk_paper_angle_grid_sensitivity as base


OUT_DIR = Path("r25_lk_no_norm_filter_bifactor_sensitivity")
L_VALUES = list(range(2, 21))
K_VALUES = list(range(10, 101, 10))
RECALL_TARGET = 0.90
L_SENSITIVITY_FIXED_K = 50
K_SENSITIVITY_FIXED_L = 20


def configure_base() -> None:
    base.OUT_DIR = OUT_DIR
    base.L_VALUES = L_VALUES
    base.K_VALUES = K_VALUES
    base.MAX_L = max(L_VALUES)
    base.MAX_K = max(K_VALUES)
    base.NORM_FILTER_RULE = no_norm.NO_NORM_FILTER_RULE
    base.LOOKUP_RULE = no_norm.NO_NORM_LOOKUP_RULE
    base.hamming_threshold_from_angle = no_norm.recall_boosted_hamming_threshold_from_angle
    base.paper_layer_bounds = no_norm.paper_layer_bounds
    base.layer_angle_from_radius = no_norm.layer_max_angle_from_radius
    base.paper_ball_query_lk = no_norm.no_norm_filter_ball_query_lk
    base.load_seed_rows = load_local_rows
    base.score_grid = score_grid
    base.best_by_dataset = optimal_by_dataset
    base.write_outputs = write_outputs

    no_norm.BOOST_END_K = max(K_VALUES)


def load_local_rows(_: str, result_path: Path) -> pd.DataFrame:
    if not result_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(result_path)
    if "norm_filter_rule" not in df.columns:
        return pd.DataFrame()
    df = df[df["norm_filter_rule"] == no_norm.NO_NORM_FILTER_RULE].copy()
    df = df[(df["L"].isin(L_VALUES)) & (df["K"].isin(K_VALUES))].copy()
    return df.sort_values(["dataset", "L", "K"]).drop_duplicates(
        ["dataset", "L", "K"],
        keep="last",
    )


def score_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["candidate_to_true_ratio"] = df["candidate_expansion_ratio"]
    df["kde_quality"] = (
        1.0 - df["kde_abs_relative_error_vs_exact_r25"]
    ).clip(lower=0.0, upper=1.0)
    df["recall_feasible"] = df["point_recall"] >= RECALL_TARGET
    scored = []

    for dataset, part in df.groupby("dataset"):
        part = part.copy()
        time_max = max(float(part["query_time_ms"].max()), 1e-12)
        expansion_max = max(float(part["candidate_to_true_ratio"].max()), 1e-12)
        part["speed_quality"] = (1.0 - part["query_time_ms"] / time_max).clip(0.0, 1.0)
        part["expansion_quality"] = (
            1.0 - part["candidate_to_true_ratio"] / expansion_max
        ).clip(0.0, 1.0)
        part["optimal_score"] = (
            0.35 * part["point_f1"]
            + 0.25 * part["point_precision"]
            + 0.20 * part["point_recall"]
            + 0.15 * part["expansion_quality"]
            + 0.05 * part["speed_quality"]
        )
        scored.append(part)

    return pd.concat(scored, ignore_index=True)


def optimal_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        feasible = part[part["recall_feasible"]].copy()
        if feasible.empty:
            source = part.copy()
            selection = "fallback_max_recall"
            sort_cols = [
                "point_recall",
                "point_f1",
                "point_precision",
                "candidate_to_true_ratio",
                "query_time_ms",
            ]
            ascending = [False, False, False, True, True]
        else:
            source = feasible
            selection = f"recall_ge_{RECALL_TARGET:.2f}_max_f1_precision"
            sort_cols = [
                "point_f1",
                "point_precision",
                "point_recall",
                "candidate_to_true_ratio",
                "query_time_ms",
            ]
            ascending = [False, False, False, True, True]

        best = source.sort_values(sort_cols, ascending=ascending).iloc[0].copy()
        best["selection"] = selection
        rows.append(best)

    return pd.DataFrame(rows)


def summarize_common(df: pd.DataFrame) -> pd.DataFrame:
    common = (
        df.groupby(["L", "K"], as_index=False)
        .agg(
            mean_precision=("point_precision", "mean"),
            mean_recall=("point_recall", "mean"),
            min_recall=("point_recall", "min"),
            mean_f1=("point_f1", "mean"),
            mean_candidate_expansion_ratio=("candidate_to_true_ratio", "mean"),
            mean_candidate_size=("candidate_size", "mean"),
            mean_query_time_ms=("query_time_ms", "mean"),
            mean_optimal_score=("optimal_score", "mean"),
        )
    )
    common["recall_feasible_all"] = common["min_recall"] >= RECALL_TARGET
    return common


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.pivot_table(index=["dataset", "K"], columns="L", values=metric, aggfunc="mean")
        .reset_index()
    )


def plot_heatmap(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    title: str,
    filename: str,
    optimum: pd.Series,
) -> None:
    figure_dir = OUT_DIR / "figures" / dataset
    figure_dir.mkdir(parents=True, exist_ok=True)
    table = (
        df[df["dataset"] == dataset]
        .pivot_table(index="K", columns="L", values=metric, aggfunc="mean")
        .reindex(index=K_VALUES, columns=L_VALUES)
    )

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 8,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(6.5, 7.2))
    image = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(L_VALUES)))
    ax.set_xticklabels(L_VALUES)
    ax.set_yticks(range(len(K_VALUES)))
    ax.set_yticklabels(K_VALUES)
    ax.set_xlabel("Hash tables L")
    ax.set_ylabel("Hash code length K")
    ax.set_title(title)

    best_x = L_VALUES.index(int(optimum["L"]))
    best_y = K_VALUES.index(int(optimum["K"]))
    ax.scatter([best_x], [best_y], marker="x", s=110, color="#d62728", linewidths=2.2)
    ax.text(
        best_x + 0.25,
        best_y + 0.15,
        "best",
        color="#d62728",
        fontsize=10,
        weight="bold",
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{filename}.png", dpi=450)
    fig.savefig(figure_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_all_heatmaps(df: pd.DataFrame, optimal: pd.DataFrame) -> None:
    specs = [
        ("point_recall", "Recall", "recall_heatmap"),
        ("point_precision", "Precision", "precision_heatmap"),
        ("point_f1", "F1", "f1_heatmap"),
        ("candidate_to_true_ratio", "Candidate expansion ratio", "candidate_expansion_heatmap"),
        ("optimal_score", "Recall-constrained precision score", "optimal_score_heatmap"),
    ]
    for dataset in base.DATASETS:
        best = optimal[optimal["dataset"] == dataset].iloc[0]
        for metric, title, filename in specs:
            plot_heatmap(df, dataset, metric, f"{dataset}: {title}", filename, best)


def plot_l_sensitivity_fixed_k(df: pd.DataFrame) -> pd.DataFrame:
    out_dir = OUT_DIR / "figures" / f"L_sensitivity_fixed_K{L_SENSITIVITY_FIXED_K}"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "amazon": "Amazon",
        "cifar10": "CIFAR-10",
        "cifar10_gist512": "CIFAR-10 GIST512",
        "isolet": "ISOLET",
    }
    fixed = df[df["K"] == L_SENSITIVITY_FIXED_K].copy()
    rows = []

    for dataset in base.DATASETS:
        part = fixed[fixed["dataset"] == dataset].sort_values("L")
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(
            part["L"],
            part["point_precision"],
            marker="o",
            markersize=3.4,
            linewidth=1.8,
            color="#4c78a8",
            label="Precision",
        )
        ax.plot(
            part["L"],
            part["point_recall"],
            marker="s",
            markersize=3.4,
            linewidth=1.8,
            color="#e15759",
            label="Recall",
        )
        ax.set_title(f"{labels.get(dataset, dataset)}: precision and recall by L at K={L_SENSITIVITY_FIXED_K}")
        ax.set_xlabel("Hash tables L")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(L_VALUES)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_precision_recall_by_L_fixed_K{L_SENSITIVITY_FIXED_K}.png", dpi=450)
        fig.savefig(out_dir / f"{dataset}_precision_recall_by_L_fixed_K{L_SENSITIVITY_FIXED_K}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(
            part["L"],
            part["candidate_to_true_ratio"],
            marker="o",
            markersize=3.4,
            linewidth=1.8,
            color="#f28e2b",
            label="Candidate expansion ratio",
        )
        ax.set_title(f"{labels.get(dataset, dataset)}: candidate expansion by L at K={L_SENSITIVITY_FIXED_K}")
        ax.set_xlabel("Hash tables L")
        ax.set_ylabel("Candidate expansion ratio")
        ax.set_xticks(L_VALUES)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_candidate_expansion_by_L_fixed_K{L_SENSITIVITY_FIXED_K}.png", dpi=450)
        fig.savefig(out_dir / f"{dataset}_candidate_expansion_by_L_fixed_K{L_SENSITIVITY_FIXED_K}.pdf")
        plt.close(fig)

        feasible = part[part["point_recall"] >= RECALL_TARGET]
        if feasible.empty:
            best = part.sort_values(
                ["point_recall", "point_f1", "point_precision", "candidate_to_true_ratio"],
                ascending=[False, False, False, True],
            ).iloc[0]
            selection = "fallback_max_recall"
        else:
            best = feasible.sort_values(
                ["point_f1", "point_precision", "point_recall", "candidate_to_true_ratio"],
                ascending=[False, False, False, True],
            ).iloc[0]
            selection = f"recall_ge_{RECALL_TARGET:.2f}_max_f1_precision"

        rows.append(
            {
                "dataset": dataset,
                "selection": selection,
                "K": int(best["K"]),
                "best_L": int(best["L"]),
                "point_precision": float(best["point_precision"]),
                "point_recall": float(best["point_recall"]),
                "point_f1": float(best["point_f1"]),
                "candidate_expansion_ratio": float(best["candidate_to_true_ratio"]),
                "candidate_size": float(best["candidate_size"]),
                "query_time_ms": float(best["query_time_ms"]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"best_L_fixed_K{L_SENSITIVITY_FIXED_K}_by_dataset.csv", index=False)
    return summary


def plot_k_sensitivity_fixed_l(df: pd.DataFrame) -> pd.DataFrame:
    out_dir = OUT_DIR / "figures" / f"K_sensitivity_fixed_L{K_SENSITIVITY_FIXED_L}"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "amazon": "Amazon",
        "cifar10": "CIFAR-10",
        "cifar10_gist512": "CIFAR-10 GIST512",
        "isolet": "ISOLET",
    }
    fixed = df[df["L"] == K_SENSITIVITY_FIXED_L].copy()
    rows = []

    for dataset in base.DATASETS:
        part = fixed[fixed["dataset"] == dataset].sort_values("K")
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(
            part["K"],
            part["point_precision"],
            marker="o",
            markersize=3.4,
            linewidth=1.8,
            color="#4c78a8",
            label="Precision",
        )
        ax.plot(
            part["K"],
            part["point_recall"],
            marker="s",
            markersize=3.4,
            linewidth=1.8,
            color="#e15759",
            label="Recall",
        )
        ax.set_title(f"{labels.get(dataset, dataset)}: precision and recall by K at L={K_SENSITIVITY_FIXED_L}")
        ax.set_xlabel("Hash code length K")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(K_VALUES)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_precision_recall_by_K_fixed_L{K_SENSITIVITY_FIXED_L}.png", dpi=450)
        fig.savefig(out_dir / f"{dataset}_precision_recall_by_K_fixed_L{K_SENSITIVITY_FIXED_L}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(
            part["K"],
            part["candidate_to_true_ratio"],
            marker="o",
            markersize=3.4,
            linewidth=1.8,
            color="#f28e2b",
            label="Candidate expansion ratio",
        )
        ax.set_title(f"{labels.get(dataset, dataset)}: candidate expansion by K at L={K_SENSITIVITY_FIXED_L}")
        ax.set_xlabel("Hash code length K")
        ax.set_ylabel("Candidate expansion ratio")
        ax.set_xticks(K_VALUES)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_candidate_expansion_by_K_fixed_L{K_SENSITIVITY_FIXED_L}.png", dpi=450)
        fig.savefig(out_dir / f"{dataset}_candidate_expansion_by_K_fixed_L{K_SENSITIVITY_FIXED_L}.pdf")
        plt.close(fig)

        feasible = part[part["point_recall"] >= RECALL_TARGET]
        if feasible.empty:
            best = part.sort_values(
                ["point_recall", "point_f1", "point_precision", "candidate_to_true_ratio"],
                ascending=[False, False, False, True],
            ).iloc[0]
            selection = "fallback_max_recall"
        else:
            best = feasible.sort_values(
                ["point_f1", "point_precision", "point_recall", "candidate_to_true_ratio"],
                ascending=[False, False, False, True],
            ).iloc[0]
            selection = f"recall_ge_{RECALL_TARGET:.2f}_max_f1_precision"

        rows.append(
            {
                "dataset": dataset,
                "selection": selection,
                "L": int(best["L"]),
                "best_K": int(best["K"]),
                "point_precision": float(best["point_precision"]),
                "point_recall": float(best["point_recall"]),
                "point_f1": float(best["point_f1"]),
                "candidate_expansion_ratio": float(best["candidate_to_true_ratio"]),
                "candidate_size": float(best["candidate_size"]),
                "query_time_ms": float(best["query_time_ms"]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"best_K_fixed_L{K_SENSITIVITY_FIXED_L}_by_dataset.csv", index=False)
    return summary


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
            elif col in {"L", "K", "best_L", "best_K"}:
                cells.append(str(int(value)))
            elif col in {"recall_feasible", "recall_feasible_all"}:
                cells.append(str(bool(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(results: pd.DataFrame, metadata: list[dict]) -> None:
    results = score_grid(results)
    results.to_csv(OUT_DIR / "r25_lk_no_norm_bifactor_all.csv", index=False)

    optimal = optimal_by_dataset(results)
    optimal.to_csv(OUT_DIR / "optimal_lk_by_dataset.csv", index=False)

    common = summarize_common(results)
    common.to_csv(OUT_DIR / "common_lk_no_norm_bifactor_summary.csv", index=False)

    for metric in [
        "point_recall",
        "point_precision",
        "point_f1",
        "candidate_to_true_ratio",
        "candidate_size",
        "query_time_ms",
        "optimal_score",
    ]:
        pivot_metric(results, metric).to_csv(OUT_DIR / f"grid_{metric}.csv", index=False)

    plot_all_heatmaps(results, optimal)
    fixed_k_l_summary = plot_l_sensitivity_fixed_k(results)
    fixed_l_k_summary = plot_k_sensitivity_fixed_l(results)

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
        "query_time_ms",
        "optimal_score",
        "recall_feasible",
    ]
    common_cols = [
        "L",
        "K",
        "mean_precision",
        "mean_recall",
        "min_recall",
        "mean_f1",
        "mean_candidate_expansion_ratio",
        "mean_query_time_ms",
        "mean_optimal_score",
        "recall_feasible_all",
    ]
    top_common = common.sort_values(
        ["recall_feasible_all", "mean_optimal_score", "mean_recall", "mean_precision"],
        ascending=[False, False, False, False],
    ).head(10)

    summary = [
        "# No-Norm L/K 双因子敏感性分析",
        "",
        "任务：同时调节哈希表数量 L 和哈希码长度 K，为每个数据集选择一组最优参数。",
        "",
        "## 实验设置",
        "",
        f"- L 网格：{', '.join(str(v) for v in L_VALUES)}。",
        f"- K 网格：{', '.join(str(v) for v in K_VALUES)}。",
        f"- 召回达标线：point_recall >= {RECALL_TARGET:.2f}。",
        "- 候选策略：禁用原模长/径向层候选过滤；径向层只用于计算角度阈值。",
        "- 最优选择：若存在召回达标组合，则在达标组合里优先最大化 F1，再看 precision、recall、候选膨胀数和查询时间；若无达标组合，则退化为最高 recall。",
        "",
        "## 每个数据集的最优 L/K",
        "",
        markdown_table(optimal, best_cols),
        "",
        f"## 固定 K={L_SENSITIVITY_FIXED_K} 的哈希表数量敏感性",
        "",
        markdown_table(
            fixed_k_l_summary,
            [
                "dataset",
                "selection",
                "K",
                "best_L",
                "point_precision",
                "point_recall",
                "point_f1",
                "candidate_expansion_ratio",
                "candidate_size",
                "query_time_ms",
            ],
        ),
        "",
        f"## 固定 L={K_SENSITIVITY_FIXED_L} 的哈希码长度敏感性",
        "",
        markdown_table(
            fixed_l_k_summary,
            [
                "dataset",
                "selection",
                "L",
                "best_K",
                "point_precision",
                "point_recall",
                "point_f1",
                "candidate_expansion_ratio",
                "candidate_size",
                "query_time_ms",
            ],
        ),
        "",
        "## 公共参数 Top 10",
        "",
        markdown_table(top_common, common_cols),
        "",
        "## 输出文件",
        "",
        "- `r25_lk_no_norm_bifactor_all.csv`",
        "- `optimal_lk_by_dataset.csv`",
        "- `common_lk_no_norm_bifactor_summary.csv`",
        "- `grid_point_recall.csv` / `grid_point_precision.csv` / `grid_point_f1.csv`",
        "- `figures/<dataset>/recall_heatmap.png`",
        "- `figures/<dataset>/precision_heatmap.png`",
        "- `figures/<dataset>/candidate_expansion_heatmap.png`",
        f"- `figures/L_sensitivity_fixed_K{L_SENSITIVITY_FIXED_K}/<dataset>_precision_recall_by_L_fixed_K{L_SENSITIVITY_FIXED_K}.png`",
        f"- `figures/K_sensitivity_fixed_L{K_SENSITIVITY_FIXED_L}/<dataset>_precision_recall_by_K_fixed_L{K_SENSITIVITY_FIXED_L}.png`",
    ]
    (OUT_DIR / "summary_zh.md").write_text("\n".join(summary), encoding="utf-8")

    config = {
        "datasets": base.DATASETS,
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "L_sensitivity_fixed_K": L_SENSITIVITY_FIXED_K,
        "K_sensitivity_fixed_L": K_SENSITIVITY_FIXED_L,
        "recall_target": RECALL_TARGET,
        "radius_mode": "query_adaptive_R25",
        "bandwidth_mode": "h(q)=0.14R25(q)",
        "norm_filter_rule": no_norm.NO_NORM_FILTER_RULE,
        "lookup_rule": no_norm.NO_NORM_LOOKUP_RULE,
        "selection": "recall >= target, then maximize F1, precision, recall, minimize candidate expansion and query time",
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
