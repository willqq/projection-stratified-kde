from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_PATH = Path("r25_lk_grid_sensitivity/r25_lk_grid_all.csv")
OUT_DIR = Path("r25_lk_grid_sensitivity/prf1_analysis")
FIG_DIR = OUT_DIR / "figures"

DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
LABELS = {
    "isolet": "ISOLET",
    "cifar10": "CIFAR-10",
    "cifar10_gist512": "CIFAR10-GIST512",
    "amazon": "Amazon",
}
METRICS = [
    ("point_precision", "Precision"),
    ("point_recall", "Recall"),
    ("point_f1", "F1"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def heatmap_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    l_values = sorted(df["L"].unique())
    k_values = sorted(df["K"].unique())
    return (
        df.pivot_table(index="K", columns="L", values=metric, aggfunc="mean")
        .reindex(index=k_values, columns=l_values)
    )


def plot_dataset_prf1(df: pd.DataFrame, dataset: str) -> None:
    part = df[df["dataset"] == dataset]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    for ax, (metric, title) in zip(axes, METRICS):
        table = heatmap_table(part, metric)
        image = ax.imshow(
            table.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("L")
        ax.set_ylabel("K")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(LABELS.get(dataset, dataset), y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{dataset}_precision_recall_f1_heatmaps.png", dpi=450)
    fig.savefig(FIG_DIR / f"{dataset}_precision_recall_f1_heatmaps.pdf")
    plt.close(fig)


def plot_common_prf1(common: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    metric_map = [
        ("mean_precision", "Mean Precision"),
        ("mean_recall", "Mean Recall"),
        ("mean_f1", "Mean F1"),
    ]
    for ax, (metric, title) in zip(axes, metric_map):
        table = heatmap_table(common, metric)
        image = ax.imshow(
            table.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("L")
        ax.set_ylabel("K")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "common_precision_recall_f1_heatmaps.png", dpi=450)
    fig.savefig(FIG_DIR / "common_precision_recall_f1_heatmaps.pdf")
    plt.close(fig)


def format_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col == "dataset":
                cells.append(str(value))
            elif col in {"L", "K"}:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df = pd.read_csv(IN_PATH)

    best_by_dataset = (
        df.sort_values(["dataset", "point_f1", "point_recall", "point_precision"], ascending=[True, False, False, False])
        .groupby("dataset", as_index=False)
        .head(1)
        .loc[
            :,
            [
                "dataset",
                "L",
                "K",
                "point_precision",
                "point_recall",
                "point_f1",
                "candidate_size",
                "query_time_ms",
            ],
        ]
    )
    best_by_dataset.to_csv(OUT_DIR / "best_f1_by_dataset.csv", index=False)

    common = (
        df.groupby(["L", "K"])
        .agg(
            mean_precision=("point_precision", "mean"),
            mean_recall=("point_recall", "mean"),
            min_recall=("point_recall", "min"),
            mean_f1=("point_f1", "mean"),
            mean_candidate_size=("candidate_size", "mean"),
            mean_query_time_ms=("query_time_ms", "mean"),
        )
        .reset_index()
    )
    common["recall_feasible_all"] = common["min_recall"] >= 0.90
    common.to_csv(OUT_DIR / "common_prf1_grid_summary.csv", index=False)

    top_mean_f1 = common.sort_values(
        ["mean_f1", "mean_recall", "mean_precision"],
        ascending=[False, False, False],
    ).head(12)
    top_mean_f1.to_csv(OUT_DIR / "top_common_mean_f1.csv", index=False)

    top_recall_feasible = common[common["recall_feasible_all"]].sort_values(
        ["mean_f1", "mean_recall", "mean_precision"],
        ascending=[False, False, False],
    ).head(12)
    top_recall_feasible.to_csv(OUT_DIR / "top_common_f1_with_recall_ge_09.csv", index=False)

    for dataset in DATASETS:
        plot_dataset_prf1(df, dataset)
    plot_common_prf1(common)

    summary_lines = [
        "# L/K Precision Recall F1 Analysis",
        "",
        "固定 `query-adaptive R25` 和 `h(q)=0.14R25(q)`，评估哈希表数量 `L` 与哈希码长度 `K` 对 Precision、Recall 和 F1 的双因子影响。",
        "",
        "## Best F1 by Dataset",
        "",
        format_markdown_table(
            best_by_dataset,
            [
                "dataset",
                "L",
                "K",
                "point_precision",
                "point_recall",
                "point_f1",
                "candidate_size",
                "query_time_ms",
            ],
        ),
        "",
        "## Top Common Settings by Mean F1",
        "",
        format_markdown_table(
            top_mean_f1,
            [
                "L",
                "K",
                "mean_precision",
                "mean_recall",
                "min_recall",
                "mean_f1",
                "mean_candidate_size",
                "mean_query_time_ms",
            ],
        ),
        "",
        "## Top Common Settings with Minimum Recall >= 0.90",
        "",
        format_markdown_table(
            top_recall_feasible,
            [
                "L",
                "K",
                "mean_precision",
                "mean_recall",
                "min_recall",
                "mean_f1",
                "mean_candidate_size",
                "mean_query_time_ms",
            ],
        ),
        "",
        "## Conclusion",
        "",
        "`K` controls the precision-recall tradeoff more strongly than `L`: larger `K` usually improves precision by shrinking hash buckets, but it reduces recall. Increasing `L` partly compensates for recall loss by adding more hash tables. F1 is maximized in a middle-to-large `K` region because it balances precision and recall, while recall-constrained settings require smaller `K`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved PR/F1 analysis to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
