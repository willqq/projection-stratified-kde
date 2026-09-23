from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("mech_common_optimal_multidataset_sensitivity_L18_local")
CSV_PATH = RESULT_DIR / "all_common_baseline_sensitivity.csv"
FIGURE_DIR = RESULT_DIR / "figures_full_metrics"

DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
COLORS = {
    "isolet": "#2F6B9A",
    "cifar10": "#2F8F83",
    "cifar10_gist512": "#D9A441",
    "amazon": "#B55252",
}
FACTOR_LABELS = {
    "L": ("Hash tables L", "L"),
    "K": ("Hash code length K", "K"),
    "delta_factor": ("Sphere interval factor", "delta / base radius"),
    "radius_percentile": ("Radius percentile", "distance percentile for R"),
    "bandwidth_factor": ("Kernel bandwidth factor", "kernel bandwidth factor"),
}
METRICS = [
    ("point_precision", "Precision"),
    ("point_recall", "Recall"),
    ("point_f1", "F1-score"),
    ("kde_abs_relative_error", "KDE relative error"),
    ("candidate_expansion_ratio", "CER"),
    ("query_time_ms", "Online query time (ms)"),
]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for factor, (title, xlabel) in FACTOR_LABELS.items():
        fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.0))
        axes = axes.ravel()
        for ax, (metric, metric_title) in zip(axes, METRICS):
            for dataset in DATASETS:
                part = df[
                    (df["dataset"] == dataset) & (df["factor"] == factor)
                ].sort_values("common_value")
                ax.plot(
                    part["common_value"].astype(float),
                    part[metric].astype(float),
                    marker="o",
                    markersize=4.0,
                    linewidth=1.8,
                    label=dataset,
                    color=COLORS[dataset],
                )
            ax.set_title(metric_title)
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.28)

        axes[0].legend(loc="best", fontsize=8, frameon=True)
        fig.suptitle(
            f"Parameter sensitivity across datasets: {title}",
            x=0.02,
            ha="left",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{factor}_dataset_comparison_full.png", dpi=260)
        fig.savefig(FIGURE_DIR / f"{factor}_dataset_comparison_full.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
