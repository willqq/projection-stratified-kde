from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("reference_sensitivity_analysis")
COMMON_DIR = Path("mech_common_optimal_multidataset_sensitivity_L18_local")
DENSE_DIR = Path("mech_requested_parameter_sensitivity_R7_dense")
RING_DIR = Path("mech_annulus_count_weighted_budget_sensitivity")
PILOT_DIR = Path("mech_annulus_pilot_neyman_decreasing_budget_2x_anchor")

METRICS = [
    "point_precision",
    "point_recall",
    "point_f1",
    "kde_abs_relative_error",
    "query_time_ms",
]

METRIC_LABELS = {
    "point_precision": "Precision",
    "point_recall": "Recall",
    "point_f1": "F1",
    "kde_abs_relative_error": "KDE error",
    "query_time_ms": "Online time",
}

DATASET_LABELS = {
    "isolet": "ISOLET",
    "cifar10": "CIFAR-10",
    "cifar10_gist512": "CIFAR-10 GIST512",
    "amazon": "Amazon",
}

FACTOR_LABELS = {
    "L": "Hash tables L",
    "K": "Hash code length K",
    "delta_factor": "Sphere interval delta/base R",
    "radius_percentile": "Query radius percentile R",
    "fixed_query_radius": "Annulus/query radius R",
    "bandwidth_factor": "Kernel bandwidth factor sigma",
    "annulus_count": "Annulus count",
    "total_sample_budget": "Total sample budget",
}


def ensure_inputs() -> None:
    required = [
        COMMON_DIR / "all_common_baseline_sensitivity.csv",
        COMMON_DIR / "common_baseline_by_dataset.csv",
        DENSE_DIR / "requested_parameter_sensitivity_dense.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required experiment outputs:\n{lines}")


def fmt_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * float(value):.{digits}f}%"


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
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
            elif isinstance(value, (bool, np.bool_)):
                cells.append(str(bool(value)))
            elif column.endswith("_pct") or column.startswith("range_"):
                cells.append(fmt_pct(value, 2))
            elif column in {"n_values", "rank"}:
                cells.append(str(int(value)))
            else:
                cells.append(fmt_float(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(frame: pd.DataFrame, columns: list[str], caption: str, label: str) -> str:
    header = " & ".join(columns) + r" \\"
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                cells.append(value.replace("_", r"\_"))
            elif column.endswith("_pct") or column.startswith("range_"):
                cells.append(fmt_pct(value, 2).replace("%", r"\%"))
            elif column in {"n_values", "rank"}:
                cells.append(str(int(value)))
            else:
                cells.append(fmt_float(value, 4))
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )


def normalize_common_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "common_value" not in df.columns:
        df["common_value"] = df["factor_value"]
    df["factor"] = df["factor"].replace(
        {"delta": "delta_factor", "fixed_query_radius": "radius_percentile"}
    )
    return df


def factor_sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_part in df.groupby("dataset"):
        dataset_part = dataset_part.copy()
        dataset_time = max(float(dataset_part["query_time_ms"].max()), 1e-12)
        for factor, part in dataset_part.groupby("factor"):
            part = part.sort_values("common_value" if "common_value" in part.columns else "factor_value")
            values = part["common_value" if "common_value" in part.columns else "factor_value"].astype(float)
            row = {
                "dataset": dataset,
                "factor": factor,
                "factor_label": FACTOR_LABELS.get(factor, factor),
                "n_values": len(part),
                "value_min": float(values.min()),
                "value_max": float(values.max()),
            }
            for metric in METRICS:
                vals = part[metric].astype(float)
                row[f"{metric}_min"] = float(vals.min())
                row[f"{metric}_max"] = float(vals.max())
                row[f"{metric}_range"] = float(vals.max() - vals.min())
                denom = 1.0 if metric != "query_time_ms" else dataset_time
                row[f"{metric}_range_pct"] = float((vals.max() - vals.min()) / max(denom, 1e-12))
            row["sensitivity_index"] = float(
                row["point_f1_range_pct"]
                + row["point_recall_range_pct"]
                + row["kde_abs_relative_error_range_pct"]
                + 0.5 * row["query_time_ms_range_pct"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_factor_summary(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(["factor", "factor_label"], as_index=False)
        .agg(
            mean_sensitivity_index=("sensitivity_index", "mean"),
            mean_f1_range=("point_f1_range", "mean"),
            mean_recall_range=("point_recall_range", "mean"),
            mean_kde_error_range=("kde_abs_relative_error_range", "mean"),
            mean_time_range_ms=("query_time_ms_range", "mean"),
            max_sensitivity_index=("sensitivity_index", "max"),
        )
        .sort_values("mean_sensitivity_index", ascending=False)
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def best_and_baseline_delta(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        base = baseline_df[baseline_df["dataset"] == dataset].iloc[0]
        for factor, factor_part in part.groupby("factor"):
            factor_part = factor_part.copy()
            factor_part["kde_quality"] = (1.0 - factor_part["kde_abs_relative_error"]).clip(0.0, 1.0)
            max_time = max(float(part["query_time_ms"].max()), 1e-12)
            factor_part["speed_quality"] = (1.0 - factor_part["query_time_ms"] / max_time).clip(0.0, 1.0)
            factor_part["local_score"] = (
                0.45 * factor_part["point_f1"]
                + 0.25 * factor_part["point_recall"]
                + 0.20 * factor_part["kde_quality"]
                + 0.10 * factor_part["speed_quality"]
            )
            best = factor_part.sort_values(
                ["local_score", "point_f1", "query_time_ms"],
                ascending=[False, False, True],
            ).iloc[0]
            baseline_factor = factor_part[
                np.isclose(
                    factor_part["common_value"].astype(float),
                    float(base[factor]) if factor in base else float(best["common_value"]),
                    rtol=1e-8,
                    atol=1e-8,
                )
            ]
            baseline = baseline_factor.iloc[0] if not baseline_factor.empty else best
            rows.append(
                {
                    "dataset": dataset,
                    "factor": factor,
                    "factor_label": FACTOR_LABELS.get(factor, factor),
                    "baseline_value": float(baseline["common_value"]),
                    "best_value": float(best["common_value"]),
                    "baseline_f1": float(baseline["point_f1"]),
                    "best_f1": float(best["point_f1"]),
                    "delta_f1": float(best["point_f1"] - baseline["point_f1"]),
                    "baseline_kde_error": float(baseline["kde_abs_relative_error"]),
                    "best_kde_error": float(best["kde_abs_relative_error"]),
                    "delta_kde_error": float(best["kde_abs_relative_error"] - baseline["kde_abs_relative_error"]),
                    "baseline_time_ms": float(baseline["query_time_ms"]),
                    "best_time_ms": float(best["query_time_ms"]),
                    "delta_time_ms": float(best["query_time_ms"] - baseline["query_time_ms"]),
                }
            )
    return pd.DataFrame(rows)


def plot_aggregate_summary(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    part = agg.sort_values("mean_sensitivity_index", ascending=True)
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"][: len(part)]
    ax.barh(part["factor_label"], part["mean_sensitivity_index"], color=colors)
    ax.set_xlabel("Mean sensitivity index")
    ax.set_title("Cross-dataset parameter sensitivity ranking")
    ax.grid(axis="x", alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_sensitivity_ranking.png", dpi=240)
    fig.savefig(out_dir / "parameter_sensitivity_ranking.pdf")
    plt.close(fig)


def plot_metric_ranges(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        summary.groupby(["factor", "factor_label"], as_index=False)
        .agg(
            f1=("point_f1_range", "mean"),
            recall=("point_recall_range", "mean"),
            kde_error=("kde_abs_relative_error_range", "mean"),
            online_ms=("query_time_ms_range", "mean"),
        )
        .sort_values("kde_error", ascending=False)
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    axes = axes.ravel()
    specs = [
        ("f1", "F1 range"),
        ("recall", "Recall range"),
        ("kde_error", "KDE error range"),
        ("online_ms", "Online time range (ms)"),
    ]
    for ax, (column, title) in zip(axes, specs):
        current = plot_df.sort_values(column, ascending=True)
        ax.barh(current["factor_label"], current[column], color="#4c78a8")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Average metric ranges across datasets", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "metric_range_by_parameter.png", dpi=240)
    fig.savefig(out_dir / "metric_range_by_parameter.pdf")
    plt.close(fig)


def plot_common_curves(df: pd.DataFrame, out_dir: Path) -> None:
    figure_dir = out_dir / "common_curves"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["point_f1", "kde_abs_relative_error", "query_time_ms"]
    colors = {
        "isolet": "#4c78a8",
        "cifar10": "#59a14f",
        "cifar10_gist512": "#f28e2b",
        "amazon": "#e15759",
    }
    for factor, part_factor in df.groupby("factor"):
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
        for ax, metric in zip(axes, metrics):
            for dataset, part in part_factor.groupby("dataset"):
                part = part.sort_values("common_value")
                ax.plot(
                    part["common_value"].astype(float),
                    part[metric].astype(float),
                    marker="o",
                    linewidth=1.8,
                    label=DATASET_LABELS.get(dataset, dataset),
                    color=colors.get(dataset),
                )
            ax.set_title(METRIC_LABELS[metric])
            ax.set_xlabel(FACTOR_LABELS.get(factor, factor))
            ax.grid(True, alpha=0.28)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Sensitivity curve: {FACTOR_LABELS.get(factor, factor)}", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{factor}_curves.png", dpi=240)
        fig.savefig(figure_dir / f"{factor}_curves.pdf")
        plt.close(fig)


def dense_isolet_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    dense = pd.read_csv(DENSE_DIR / "requested_parameter_sensitivity_dense.csv")
    dense = dense.rename(columns={"factor_value": "common_value"})
    dense["factor"] = dense["factor"].replace({"delta": "delta_factor"})
    dense["factor_label"] = dense["factor"].map(FACTOR_LABELS).fillna(dense["factor"])
    summary = factor_sensitivity_summary(dense.assign(dataset="isolet_dense"))
    best = []
    for factor, part in dense.groupby("factor"):
        candidates = part.copy()
        candidates["kde_quality"] = (1.0 - candidates["kde_abs_relative_error"]).clip(0.0, 1.0)
        max_time = max(float(candidates["query_time_ms"].max()), 1e-12)
        candidates["speed_quality"] = (1.0 - candidates["query_time_ms"] / max_time).clip(0.0, 1.0)
        candidates["score"] = (
            0.45 * candidates["point_f1"]
            + 0.25 * candidates["point_recall"]
            + 0.20 * candidates["kde_quality"]
            + 0.10 * candidates["speed_quality"]
        )
        row = candidates.sort_values(["score", "point_f1"], ascending=[False, False]).iloc[0]
        best.append(
            {
                "factor": factor,
                "factor_label": FACTOR_LABELS.get(factor, factor),
                "best_value": float(row["common_value"]),
                "point_precision": float(row["point_precision"]),
                "point_recall": float(row["point_recall"]),
                "point_f1": float(row["point_f1"]),
                "kde_abs_relative_error": float(row["kde_abs_relative_error"]),
                "query_time_ms": float(row["query_time_ms"]),
            }
        )
    return summary, pd.DataFrame(best)


def ring_budget_summary() -> pd.DataFrame:
    path = RING_DIR / "all_annulus_count_weighted_budget.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for dataset, part in df.groupby("dataset"):
        baseline = part[
            (part["annulus_count"] == 1)
            & (part["total_sample_budget"] == 16)
        ]
        if baseline.empty:
            baseline = part.sort_values("score_kde", ascending=False).head(1)
        baseline = baseline.iloc[0]
        best = part.sort_values("score_kde", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "baseline_annulus_count": int(baseline["annulus_count"]),
                "baseline_total_sample_budget": int(baseline["total_sample_budget"]),
                "best_annulus_count": int(best["annulus_count"]),
                "best_total_sample_budget": int(best["total_sample_budget"]),
                "baseline_kde_error": float(baseline["kde_abs_relative_error"]),
                "best_kde_error": float(best["kde_abs_relative_error"]),
                "kde_error_reduction_pct": float(
                    (baseline["kde_abs_relative_error"] - best["kde_abs_relative_error"])
                    / max(float(baseline["kde_abs_relative_error"]), 1e-12)
                ),
                "baseline_query_time_ms": float(baseline["query_time_ms"]),
                "best_query_time_ms": float(best["query_time_ms"]),
            }
        )
    return pd.DataFrame(rows)


def pilot_decreasing_summary() -> pd.DataFrame:
    path = PILOT_DIR / "pilot_neyman_decreasing_budget.csv"
    if not path.exists():
        path = PILOT_DIR / "all_decreasing_budget.csv"
    if not path.exists():
        candidates = list(PILOT_DIR.glob("*.csv"))
        if not candidates:
            return pd.DataFrame()
        path = candidates[0]
    df = pd.read_csv(path)
    required = {"dataset", "annulus_count", "total_sample_budget", "kde_abs_relative_error", "query_time_ms"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for dataset, part in df.groupby("dataset"):
        baseline = part[
            (part["annulus_count"] == 2)
            & (part["total_sample_budget"] == 500)
        ]
        if baseline.empty:
            baseline = part.sort_values("annulus_count").head(1)
        baseline = baseline.iloc[0]
        large = part[part["annulus_count"] >= 8]
        if large.empty:
            large = part
        best = large.sort_values(["kde_abs_relative_error", "query_time_ms"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "baseline_annulus_count": int(baseline["annulus_count"]),
                "baseline_total_sample_budget": int(baseline["total_sample_budget"]),
                "best_large_annulus_count": int(best["annulus_count"]),
                "best_large_total_sample_budget": int(best["total_sample_budget"]),
                "baseline_kde_error": float(baseline["kde_abs_relative_error"]),
                "best_large_kde_error": float(best["kde_abs_relative_error"]),
                "error_ratio_vs_baseline": float(best["kde_abs_relative_error"] / max(float(baseline["kde_abs_relative_error"]), 1e-12)),
                "baseline_query_time_ms": float(baseline["query_time_ms"]),
                "best_large_query_time_ms": float(best["query_time_ms"]),
                "time_ratio_vs_baseline": float(best["query_time_ms"] / max(float(baseline["query_time_ms"]), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    common_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    per_dataset_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    best_delta: pd.DataFrame,
    dense_best: pd.DataFrame,
    ring_summary: pd.DataFrame,
    pilot_summary: pd.DataFrame,
) -> None:
    top = aggregate_summary.sort_values("mean_sensitivity_index", ascending=False)
    baseline_cols = [
        "dataset",
        "L",
        "K",
        "delta_factor",
        "radius_percentile",
        "bandwidth_factor",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error",
        "query_time_ms",
    ]
    base_show = baseline_df[baseline_cols].copy()
    base_show["dataset"] = base_show["dataset"].map(DATASET_LABELS).fillna(base_show["dataset"])

    agg_show = top[
        [
            "rank",
            "factor_label",
            "mean_sensitivity_index",
            "mean_f1_range",
            "mean_recall_range",
            "mean_kde_error_range",
            "mean_time_range_ms",
        ]
    ].copy()

    dense_show = dense_best[
        [
            "factor_label",
            "best_value",
            "point_precision",
            "point_recall",
            "point_f1",
            "kde_abs_relative_error",
            "query_time_ms",
        ]
    ].copy()

    best_show = best_delta[
        [
            "dataset",
            "factor_label",
            "baseline_value",
            "best_value",
            "delta_f1",
            "delta_kde_error",
            "delta_time_ms",
        ]
    ].copy()
    best_show["dataset"] = best_show["dataset"].map(DATASET_LABELS).fillna(best_show["dataset"])

    lines = [
        "# 参考原实验的敏感性分析",
        "",
        "本分析不重新训练模型，而是复用原实验输出，保持原来的数据划分、MECH 训练设置和单因素扰动方式。主要参考 `mech_common_optimal_multidataset_sensitivity_L18_local` 的四数据集结果，并用 `mech_requested_parameter_sensitivity_R7_dense` 的 ISOLET R=7 密集单因素结果交叉验证结论。",
        "",
        "## 基准设置",
        "",
        "四数据集局部敏感性采用公共基准：L=12，K=5，delta_factor=0.05，radius_percentile=70，bandwidth_factor=0.25，hamming_probe=2，min_collisions=2，annulus_count=1，ring_sample_size=16。",
        "",
        markdown_table(base_show, baseline_cols),
        "",
        "## 跨数据集敏感度排序",
        "",
        "敏感度指数由 F1 变化幅度、Recall 变化幅度、KDE 相对误差变化幅度和在线时间变化幅度组成，其中在线时间权重为 0.5。数值越大，表示该参数对整体行为影响越强。",
        "",
        markdown_table(agg_show, list(agg_show.columns)),
        "",
        "结论上，`radius_percentile` 和 `bandwidth_factor` 是最敏感的两个参数：前者改变查询球/近似环覆盖范围，直接影响 Precision-Recall 权衡；后者不改变候选集合，但会显著改变核权重和 KDE 误差。`K` 对候选规模和召回率的影响次之；`L` 与 `delta_factor` 在公共基准附近整体较稳定。",
        "",
        "## 各参数相对基准的最佳单因素点",
        "",
        "下表给出每个数据集、每个参数单独扰动时按综合分数选择的最佳点，以及相对公共基准的变化。`delta_kde_error < 0` 表示 KDE 误差下降。",
        "",
        markdown_table(best_show, list(best_show.columns)),
        "",
        "## ISOLET R=7 密集单因素交叉验证",
        "",
        "R=7 密集实验显示，L 增大时 Recall 从低值逐步升高，但候选扩张和耗时增加；K 在 7--8 附近取得较好的 Recall/F1 折中；固定候选集合下，sigma 对 KDE 误差最明显，0.25--0.5 优于较大的带宽因子。",
        "",
        markdown_table(dense_show, list(dense_show.columns)),
        "",
    ]

    if not ring_summary.empty:
        ring_show = ring_summary.copy()
        ring_show["dataset"] = ring_show["dataset"].map(DATASET_LABELS).fillna(ring_show["dataset"])
        lines.extend(
            [
                "## 近似环数量与总采样预算",
                "",
                "参考原有 strict weighted total-budget 实验，增加总采样预算通常能降低 KDE 误差；在固定公共检索参数下，annulus_count 本身不改变点检索 P/R/F1，主要影响 KDE 采样分配和在线时间。",
                "",
                markdown_table(ring_show, list(ring_show.columns)),
                "",
            ]
        )

    if not pilot_summary.empty:
        pilot_show = pilot_summary.copy()
        pilot_show["dataset"] = pilot_show["dataset"].map(DATASET_LABELS).fillna(pilot_show["dataset"])
        lines.extend(
            [
                "## 递减预算采样补充结果",
                "",
                "Pilot-Neyman 递减预算实验说明，较大的环数配合更小的总预算可以降低采样开销，但误差并非单调下降；Amazon 和 GIST512 上大环数仍可接近基准误差，而 CIFAR-10 与 ISOLET 上误差放大更明显。",
                "",
                markdown_table(pilot_show, list(pilot_show.columns)),
                "",
            ]
        )

    lines.extend(
        [
            "## 论文写法建议",
            "",
            "1. 参数敏感性表中优先展示 L、K、delta、R/radius_percentile、sigma 五个参数，主文强调 R 与 sigma，附录给完整曲线。",
            "2. 对 L 和 delta 的描述应避免声称完全不敏感，只能说在公共基准邻域内变化较小；在 R=7 密集实验中 L 对召回率仍有明显影响。",
            "3. 对 annulus_count 与 sample budget 应单独作为 KDE 采样敏感性分析，不要和哈希检索参数混在同一张主表中。",
            "4. 若论文篇幅有限，主文使用 `parameter_sensitivity_ranking.png` 和综合排序表，附录放 `common_curves/*.png`。",
            "",
            "## 输出文件",
            "",
            "- `parameter_sensitivity_ranking.png` / `.pdf`",
            "- `metric_range_by_parameter.png` / `.pdf`",
            "- `common_curves/<factor>_curves.png` / `.pdf`",
            "- `cross_dataset_factor_sensitivity.csv`",
            "- `aggregate_factor_sensitivity.csv`",
            "- `best_vs_baseline_by_dataset_factor.csv`",
            "- `latex_sensitivity_section.tex`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    latex_lines = [
        r"\subsection{Parameter Sensitivity Analysis}",
        "",
        "Following the original one-factor-at-a-time protocol, we fix the common MECH baseline and perturb one parameter while keeping the remaining parameters unchanged. The analysis uses the same train/index/query split and the same online query-time definition as the main experiments.",
        "",
        latex_table(
            agg_show,
            list(agg_show.columns),
            "Cross-dataset sensitivity ranking of the main MECH parameters.",
            "tab:reference_sensitivity_ranking",
        ),
        "",
        "The results show that the query-radius percentile and the kernel bandwidth factor are the most sensitive parameters. The radius controls the queried annulus region and therefore changes the precision--recall trade-off, whereas the bandwidth changes only the kernel weights and mainly affects KDE error. The hash code length is also important because it controls candidate expansion and recall. In contrast, the number of hash tables and the sphere interval factor are relatively stable around the selected common baseline.",
    ]
    (out_dir / "latex_sensitivity_section.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def main() -> None:
    ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    common_df = normalize_common_results(pd.read_csv(COMMON_DIR / "all_common_baseline_sensitivity.csv"))
    baseline_df = pd.read_csv(COMMON_DIR / "common_baseline_by_dataset.csv")
    per_dataset_summary = factor_sensitivity_summary(common_df)
    aggregate_summary = aggregate_factor_summary(per_dataset_summary)
    best_delta = best_and_baseline_delta(common_df, baseline_df)
    dense_summary, dense_best = dense_isolet_summary()
    ring_summary = ring_budget_summary()
    pilot_summary = pilot_decreasing_summary()

    common_df.to_csv(OUT_DIR / "source_common_sensitivity_normalized.csv", index=False)
    per_dataset_summary.to_csv(OUT_DIR / "cross_dataset_factor_sensitivity.csv", index=False)
    aggregate_summary.to_csv(OUT_DIR / "aggregate_factor_sensitivity.csv", index=False)
    best_delta.to_csv(OUT_DIR / "best_vs_baseline_by_dataset_factor.csv", index=False)
    dense_summary.to_csv(OUT_DIR / "isolet_dense_factor_sensitivity.csv", index=False)
    dense_best.to_csv(OUT_DIR / "isolet_dense_best_by_factor.csv", index=False)
    if not ring_summary.empty:
        ring_summary.to_csv(OUT_DIR / "ring_budget_sensitivity_summary.csv", index=False)
    if not pilot_summary.empty:
        pilot_summary.to_csv(OUT_DIR / "pilot_neyman_decreasing_budget_summary.csv", index=False)

    plot_aggregate_summary(aggregate_summary, OUT_DIR)
    plot_metric_ranges(per_dataset_summary, OUT_DIR)
    plot_common_curves(common_df, OUT_DIR)

    config = {
        "source_common_dir": str(COMMON_DIR),
        "source_dense_dir": str(DENSE_DIR),
        "source_ring_dir": str(RING_DIR),
        "source_pilot_dir": str(PILOT_DIR),
        "metrics": METRICS,
        "sensitivity_index": "F1 range + recall range + KDE error range + 0.5 * normalized online-time range",
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_report(
        OUT_DIR,
        common_df,
        baseline_df,
        per_dataset_summary,
        aggregate_summary,
        best_delta,
        dense_best,
        ring_summary,
        pilot_summary,
    )
    print(f"Saved reference sensitivity analysis to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
