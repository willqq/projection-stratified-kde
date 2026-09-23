from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    "mech_delta_sample_size_bifactor_sensitivity_s512/all_delta_sample_bifactor.csv"
)
METRIC = "kde_abs_relative_error"
SAMPLE_COLUMN = "ring_sample_size"
INTERVAL_COLUMN = "divisions"
DATASET_COLUMN = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw one KDE error boxplot for each ring sample size. "
            "Boxes compare annulus interval settings."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="CSV containing delta/sample-size bifactor sensitivity results.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG/PDF outputs. Defaults to <input parent>/figures/boxplots_by_sample_size.",
    )
    parser.add_argument(
        "--metric",
        default=METRIC,
        help="Error metric column to plot.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only save PNG files.",
    )
    return parser.parse_args()


def numeric_sorted(values: pd.Series) -> list[float]:
    return sorted(values.dropna().unique(), key=float)


def dataset_offsets(datasets: list[str]) -> dict[str, float]:
    if len(datasets) <= 1:
        return {datasets[0]: 0.0} if datasets else {}
    offsets = np.linspace(-0.16, 0.16, len(datasets))
    return {dataset: float(offset) for dataset, offset in zip(datasets, offsets)}


def plot_one_sample_size(
    df: pd.DataFrame,
    sample_size: float,
    intervals: list[float],
    datasets: list[str],
    output_dir: Path,
    metric: str,
    y_limit: tuple[float, float],
    save_pdf: bool,
) -> None:
    part = df[df[SAMPLE_COLUMN] == sample_size].copy()
    values = [
        part.loc[part[INTERVAL_COLUMN] == interval, metric].dropna().to_numpy(dtype=float)
        for interval in intervals
    ]
    positions = np.arange(1, len(intervals) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#c2410c",
            "markeredgecolor": "#7c2d12",
            "markersize": 4.5,
        },
        medianprops={"color": "#111827", "linewidth": 1.5},
        boxprops={"linewidth": 1.15, "color": "#374151"},
        whiskerprops={"linewidth": 1.05, "color": "#4b5563"},
        capprops={"linewidth": 1.05, "color": "#4b5563"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#f97316",
            "markeredgecolor": "#9a3412",
            "markersize": 4,
            "alpha": 0.65,
        },
    )
    for box in boxes["boxes"]:
        box.set_facecolor("#bfdbfe")
        box.set_alpha(0.78)

    colors = {
        dataset: color
        for dataset, color in zip(
            datasets,
            ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#0891b2", "#ca8a04"],
        )
    }
    offsets = dataset_offsets(datasets)
    for dataset in datasets:
        dataset_part = part[part[DATASET_COLUMN] == dataset]
        x = []
        y = []
        for index, interval in enumerate(intervals, start=1):
            points = dataset_part.loc[
                dataset_part[INTERVAL_COLUMN] == interval,
                metric,
            ].dropna()
            x.extend([index + offsets.get(dataset, 0.0)] * len(points))
            y.extend(points.astype(float).tolist())
        ax.scatter(
            x,
            y,
            s=32,
            color=colors[dataset],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.45,
            label=dataset,
            zorder=3,
        )

    ax.set_title(f"KDE error by annulus interval, sample size = {int(sample_size)}")
    ax.set_xlabel("Annulus interval setting: divisions m (larger m means smaller delta)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(interval)) for interval in intervals])
    ax.set_ylim(*y_limit)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(
        title="Dataset",
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=8.5,
        title_fontsize=8.5,
    )

    fig.tight_layout()
    stem = f"kde_error_boxplot_sample_{int(sample_size)}"
    fig.savefig(output_dir / f"{stem}.png", dpi=260)
    if save_pdf:
        fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_path.parent / "figures" / "boxplots_by_sample_size"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {args.metric, SAMPLE_COLUMN, INTERVAL_COLUMN, DATASET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    df = df.dropna(subset=[args.metric, SAMPLE_COLUMN, INTERVAL_COLUMN, DATASET_COLUMN])
    intervals = numeric_sorted(df[INTERVAL_COLUMN])
    sample_sizes = numeric_sorted(df[SAMPLE_COLUMN])
    datasets = sorted(df[DATASET_COLUMN].astype(str).unique())
    df[DATASET_COLUMN] = df[DATASET_COLUMN].astype(str)

    y_values = df[args.metric].astype(float)
    y_min = min(0.0, float(y_values.min()))
    y_max = float(y_values.max())
    padding = max((y_max - y_min) * 0.08, 0.02)
    y_limit = (y_min, y_max + padding)

    for sample_size in sample_sizes:
        plot_one_sample_size(
            df,
            sample_size,
            intervals,
            datasets,
            output_dir,
            args.metric,
            y_limit,
            save_pdf=not args.no_pdf,
        )

    print(f"Saved {len(sample_sizes)} boxplot figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
