from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("r25_lk_2_20_k10_100_bifactor_sensitivity")
INPUT = OUT_DIR / "r25_lk_paper_angle_grid_all.csv"
RECALL_FLOOR_RATIO = 0.50
FORCED_MIN_K_VALUES = [20, 30]


def add_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["k_quality"] = (df["K"].astype(float) - 10.0) / 90.0
    df["kde_quality"] = (
        1.0 - df["kde_abs_relative_error_vs_exact_r25"].astype(float)
    ).clip(lower=0.0, upper=1.0)
    df["large_k_recall_score"] = (
        0.55 * df["point_recall"].astype(float)
        + 0.20 * df["k_quality"]
        + 0.15 * df["kde_quality"]
        + 0.10 * df["point_f1"].astype(float)
    )
    return df


def select_largest_k_above_recall_floor(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        best_recall = float(part["point_recall"].max())
        floor = ratio * best_recall
        eligible = part[part["point_recall"] >= floor].copy()
        pick = eligible.sort_values(
            [
                "K",
                "point_recall",
                "kde_abs_relative_error_vs_exact_r25",
                "large_k_recall_score",
                "candidate_size",
            ],
            ascending=[False, False, True, False, True],
        ).iloc[0].copy()
        pick["selection_rule"] = f"largest K with recall >= {ratio:.2f} * dataset_best_recall"
        pick["dataset_best_recall"] = best_recall
        pick["recall_floor"] = floor
        pick["recall_retention"] = float(pick["point_recall"]) / max(best_recall, 1e-12)
        rows.append(pick)
    return pd.DataFrame(rows)


def select_best_recall_with_min_k(df: pd.DataFrame, min_k: int) -> pd.DataFrame:
    rows = []
    for dataset, part in df[df["K"] >= min_k].groupby("dataset"):
        best_recall = float(part["point_recall"].max())
        pick = part.sort_values(
            [
                "point_recall",
                "K",
                "kde_abs_relative_error_vs_exact_r25",
                "large_k_recall_score",
                "candidate_size",
            ],
            ascending=[False, False, True, False, True],
        ).iloc[0].copy()
        pick["selection_rule"] = f"best recall with K >= {min_k}"
        pick["dataset_best_recall_in_forced_range"] = best_recall
        rows.append(pick)
    return pd.DataFrame(rows)


def summarize_common(df: pd.DataFrame) -> pd.DataFrame:
    common = (
        df.groupby(["L", "K"], as_index=False)
        .agg(
            mean_precision=("point_precision", "mean"),
            mean_recall=("point_recall", "mean"),
            min_recall=("point_recall", "min"),
            mean_f1=("point_f1", "mean"),
            mean_kde_error=("kde_abs_relative_error_vs_exact_r25", "mean"),
            max_kde_error=("kde_abs_relative_error_vs_exact_r25", "max"),
            mean_candidate_size=("candidate_size", "mean"),
            mean_query_time_ms=("query_time_ms", "mean"),
            mean_large_k_recall_score=("large_k_recall_score", "mean"),
        )
    )
    common["k_quality"] = (common["K"].astype(float) - 10.0) / 90.0
    return common


def common_largest_k_above_floor(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    dataset_best = df.groupby("dataset")["point_recall"].max().rename("best_recall")
    work = df.merge(dataset_best, on="dataset")
    work["passes_floor"] = work["point_recall"] >= ratio * work["best_recall"]
    grouped = (
        work.groupby(["L", "K"], as_index=False)
        .agg(
            pass_count=("passes_floor", "sum"),
            mean_recall=("point_recall", "mean"),
            min_recall=("point_recall", "min"),
            mean_kde_error=("kde_abs_relative_error_vs_exact_r25", "mean"),
            mean_candidate_size=("candidate_size", "mean"),
            mean_query_time_ms=("query_time_ms", "mean"),
            mean_large_k_recall_score=("large_k_recall_score", "mean"),
        )
    )
    grouped["required_pass_count"] = int(df["dataset"].nunique())
    candidates = grouped[grouped["pass_count"] == grouped["required_pass_count"]]
    if candidates.empty:
        candidates = grouped
    return candidates.sort_values(
        ["pass_count", "K", "mean_recall", "mean_kde_error", "mean_query_time_ms"],
        ascending=[False, False, False, True, True],
    )


def common_best_recall_with_min_k(df: pd.DataFrame, min_k: int) -> pd.DataFrame:
    common = summarize_common(df)
    return common[common["K"] >= min_k].sort_values(
        ["mean_recall", "K", "mean_kde_error", "mean_query_time_ms"],
        ascending=[False, False, True, True],
    )


def fmt(value: object, column: str) -> str:
    if column in {"dataset", "selection_rule"}:
        return str(value)
    if column in {"L", "K", "forced_min_K", "pass_count", "required_pass_count"}:
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return f"{float(value):.4f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[col], col) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    df = add_quality_columns(pd.read_csv(INPUT))
    df.to_csv(OUT_DIR / "r25_lk_paper_angle_grid_all_large_k_scored.csv", index=False)

    recall_floor_selection = select_largest_k_above_recall_floor(df, RECALL_FLOOR_RATIO)
    recall_floor_selection.to_csv(
        OUT_DIR / "large_k_recall_floor_selection_by_dataset.csv",
        index=False,
    )

    forced_frames = []
    for min_k in FORCED_MIN_K_VALUES:
        forced = select_best_recall_with_min_k(df, min_k)
        forced["forced_min_K"] = min_k
        forced_frames.append(forced)
    forced_selection = pd.concat(forced_frames, ignore_index=True)
    forced_selection.to_csv(
        OUT_DIR / "forced_min_k_best_recall_by_dataset.csv",
        index=False,
    )

    common = summarize_common(df)
    common.to_csv(OUT_DIR / "common_lk_large_k_recall_summary.csv", index=False)

    common_floor = common_largest_k_above_floor(df, RECALL_FLOOR_RATIO)
    common_floor.to_csv(OUT_DIR / "common_large_k_recall_floor_candidates.csv", index=False)

    common_forced_frames = []
    for min_k in FORCED_MIN_K_VALUES:
        forced_common = common_best_recall_with_min_k(df, min_k).copy()
        forced_common["forced_min_K"] = min_k
        common_forced_frames.append(forced_common.head(10))
    pd.concat(common_forced_frames, ignore_index=True).to_csv(
        OUT_DIR / "common_forced_min_k_best_recall.csv",
        index=False,
    )

    dataset_cols = [
        "dataset",
        "selection_rule",
        "L",
        "K",
        "point_precision",
        "point_recall",
        "recall_retention",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "query_time_ms",
        "large_k_recall_score",
    ]
    forced_cols = [
        "dataset",
        "selection_rule",
        "L",
        "K",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "query_time_ms",
        "large_k_recall_score",
    ]
    common_cols = [
        "L",
        "K",
        "pass_count",
        "required_pass_count",
        "mean_recall",
        "min_recall",
        "mean_kde_error",
        "mean_candidate_size",
        "mean_query_time_ms",
        "mean_large_k_recall_score",
    ]
    common_forced_cols = [
        "forced_min_K",
        "L",
        "K",
        "mean_recall",
        "min_recall",
        "mean_kde_error",
        "mean_candidate_size",
        "mean_query_time_ms",
        "mean_large_k_recall_score",
    ]

    summary = [
        "# 大 K 优先的 L/K 选择结果",
        "",
        "本文件基于已完成的 `L=2..20, K=10..100` 双因子结果重新选参，没有重跑实验点。",
        "",
        "## 选参规则",
        "",
        f"- 保召回大 K 规则：对每个数据集，先找到该数据集在网格内的最佳召回，再选择召回不低于最佳召回 `{RECALL_FLOOR_RATIO:.2f}` 倍的候选；在候选中优先选择更大的 `K`。",
        "- 强制大 K 规则：分别在 `K>=20` 和 `K>=30` 的范围内，选择召回最高的组合；若召回相同，则选择更大的 `K`。",
        "- 这样可以避免单纯为了大 K 选到 `K=100` 但召回极低的组合。",
        "",
        "## 保召回大 K：各数据集推荐",
        "",
        markdown_table(recall_floor_selection, dataset_cols),
        "",
        "## 强制 K>=20 / K>=30：各数据集最高召回",
        "",
        markdown_table(forced_selection, forced_cols),
        "",
        "## 公共组合：保召回大 K 候选",
        "",
        markdown_table(common_floor.head(10), common_cols),
        "",
        "说明：`pass_count` 表示该公共 `(L,K)` 组合有多少个数据集达到各自最佳召回的 50%。如果没有组合四个数据集全部通过，则按通过数量、K、平均召回排序。",
        "",
        "## 公共组合：强制 K>=20 / K>=30 的最高平均召回",
        "",
        markdown_table(
            pd.concat(common_forced_frames, ignore_index=True),
            common_forced_cols,
        ),
        "",
        "## 输出文件",
        "",
        "- `r25_lk_paper_angle_grid_all_large_k_scored.csv`",
        "- `large_k_recall_floor_selection_by_dataset.csv`",
        "- `forced_min_k_best_recall_by_dataset.csv`",
        "- `common_lk_large_k_recall_summary.csv`",
        "- `common_large_k_recall_floor_candidates.csv`",
        "- `common_forced_min_k_best_recall.csv`",
    ]
    (OUT_DIR / "summary_large_k_zh.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved large-K recall selection outputs to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
