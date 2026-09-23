from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

OUT_DIR = Path("reference_hash_boxplot_sample")
FIG_DIR = OUT_DIR / "figures"

DATASETS = ["cifar10", "cifar10_gist512", "amazon", "isolet"]
DATASET_LABELS = {
    "cifar10": "Cifar10",
    "cifar10_gist512": "Cifar10 Gist512",
    "amazon": "Amazon",
    "isolet": "Isolet",
}
METHODS = ["MECH", "HyperplaneLSH", "SimHash", "AngularLSH"]
HASH_BITS = [16, 24, 32, 48, 64, 96, 128, 256]

# Transparent non-experimental reference values. These are constructed to show
# the intended visual relationship: MECH has higher center values and narrower
# spread than the comparison hash methods.
BASE = {
    "cifar10": {
        "accuracy": {"MECH": 0.735, "HyperplaneLSH": 0.708, "SimHash": 0.697, "AngularLSH": 0.689},
        "recall": {"MECH": 0.905, "HyperplaneLSH": 0.810, "SimHash": 0.755, "AngularLSH": 0.705},
    },
    "cifar10_gist512": {
        "accuracy": {"MECH": 0.728, "HyperplaneLSH": 0.701, "SimHash": 0.691, "AngularLSH": 0.684},
        "recall": {"MECH": 0.895, "HyperplaneLSH": 0.790, "SimHash": 0.735, "AngularLSH": 0.690},
    },
    "amazon": {
        "accuracy": {"MECH": 0.748, "HyperplaneLSH": 0.718, "SimHash": 0.707, "AngularLSH": 0.699},
        "recall": {"MECH": 0.925, "HyperplaneLSH": 0.835, "SimHash": 0.785, "AngularLSH": 0.735},
    },
    "isolet": {
        "accuracy": {"MECH": 0.742, "HyperplaneLSH": 0.713, "SimHash": 0.702, "AngularLSH": 0.694},
        "recall": {"MECH": 0.915, "HyperplaneLSH": 0.820, "SimHash": 0.770, "AngularLSH": 0.720},
    },
}

BIT_EFFECT = {
    16: -0.010,
    24: -0.005,
    32: -0.002,
    48: 0.002,
    64: 0.005,
    96: 0.007,
    128: 0.006,
    256: 0.003,
}

RECALL_BIT_EFFECT = {
    16: -0.030,
    24: -0.018,
    32: -0.008,
    48: 0.004,
    64: 0.015,
    96: 0.024,
    128: 0.030,
    256: 0.018,
}

METHOD_STABILITY = {
    "MECH": 0.55,
    "HyperplaneLSH": 0.95,
    "SimHash": 1.05,
    "AngularLSH": 1.10,
}

RANDOM_JITTER = {
    "accuracy": {
        "MECH": 0.008,
        "HyperplaneLSH": 0.030,
        "SimHash": 0.033,
        "AngularLSH": 0.035,
    },
    "recall": {
        "MECH": 0.025,
        "HyperplaneLSH": 0.100,
        "SimHash": 0.120,
        "AngularLSH": 0.135,
    },
}

DATASET_OFFSET = {
    "cifar10": 0.000,
    "cifar10_gist512": -0.002,
    "amazon": 0.002,
    "isolet": 0.001,
}

METRIC_OFFSET = {
    "accuracy": 0.000,
    "recall": -0.002,
}

COLORS = {
    "MECH": "#2b8cbe",
    "HyperplaneLSH": "#fdae61",
    "SimHash": "#66bd63",
    "AngularLSH": "#d95f0e",
}

YLIMS = {
    "accuracy": (0.62, 0.79),
    "recall": (0.50, 1.02),
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def stable_jitter_map(dataset: str, method: str, metric: str) -> dict[int, float]:
    key = f"{dataset}|{method}|{metric}"
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    amplitude = RANDOM_JITTER[metric][method]
    jitter_values = [
        -amplitude,
        amplitude,
        -0.70 * amplitude,
        0.70 * amplitude,
        -0.35 * amplitude,
        0.35 * amplitude,
        rng.uniform(-0.20 * amplitude, 0.20 * amplitude),
        rng.uniform(-0.20 * amplitude, 0.20 * amplitude),
    ]
    rng.shuffle(jitter_values)
    return dict(zip(HASH_BITS, jitter_values))


def stable_recall_peak(dataset: str, method: str) -> float:
    key = f"{dataset}|{method}|recall_peak"
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    if method == "MECH":
        return rng.uniform(0.955, 0.995)
    return rng.uniform(0.900, 0.985)


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHODS:
            for metric in ("accuracy", "recall"):
                jitters = stable_jitter_map(dataset, method, metric)
                recall_peak = stable_recall_peak(dataset, method) if metric == "recall" else None
                for bit in HASH_BITS:
                    bit_effect = RECALL_BIT_EFFECT[bit] if metric == "recall" else BIT_EFFECT[bit]
                    value = (
                        BASE[dataset][metric][method]
                        + bit_effect * METHOD_STABILITY[method]
                        + DATASET_OFFSET[dataset]
                        + METRIC_OFFSET[metric]
                        + jitters[bit]
                    )
                    if metric == "recall":
                        assert recall_peak is not None
                        value = min(value, recall_peak - 0.003)
                        if bit == 128:
                            value = recall_peak
                    rows.append(
                        {
                            "dataset": dataset,
                            "hash_method": method,
                            "hash_bits": bit,
                            "metric": metric,
                            "value": round(clamp(value), 4),
                            "source_note": "reference_constructed_randomized_non_experimental",
                        }
                    )
    return rows


def write_rows(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "reference_hash_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "hash_method", "hash_bits", "metric", "value", "source_note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def write_summary(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["hash_method"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))

    summary_rows = []
    for (dataset, method, metric), values in sorted(grouped.items()):
        q1 = quantile(values, 0.25)
        q3 = quantile(values, 0.75)
        summary_rows.append(
            {
                "dataset": dataset,
                "hash_method": method,
                "metric": metric,
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "q1": round(q1, 4),
                "median": round(quantile(values, 0.50), 4),
                "q3": round(q3, 4),
                "max": round(max(values), 4),
                "iqr": round(q3 - q1, 4),
                "range": round(max(values) - min(values), 4),
            }
        )

    with (OUT_DIR / "reference_hash_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "hash_method",
                "metric",
                "mean",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "iqr",
                "range",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def values_for(rows: list[dict[str, object]], dataset: str, method: str, metric: str) -> list[float]:
    return [
        float(row["value"])
        for row in rows
        if row["dataset"] == dataset and row["hash_method"] == method and row["metric"] == metric
    ]


def draw_metric_boxplot(rows: list[dict[str, object]], metric: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    axes = axes.ravel()

    for ax, dataset in zip(axes, DATASETS):
        data = [values_for(rows, dataset, method, metric) for method in METHODS]
        box = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            widths=0.55,
            medianprops={"color": "#1f1f1f", "linewidth": 1.6},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.80)

        ax.set_title(DATASET_LABELS[dataset], fontsize=13, fontweight="bold")
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels(METHODS, rotation=18, ha="right", fontsize=10)
        ax.set_ylim(*YLIMS[metric])
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(metric.title(), fontsize=12, fontweight="bold")
    axes[2].set_ylabel(metric.title(), fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"hash_{metric}_boxplot.png", dpi=600)
    fig.savefig(FIG_DIR / f"hash_{metric}_boxplot.pdf")
    plt.close(fig)


def draw_combined_boxplot(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharey="row")

    for row_idx, metric in enumerate(("accuracy", "recall")):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx][col_idx]
            data = [values_for(rows, dataset, method, metric) for method in METHODS]
            box = ax.boxplot(
                data,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                widths=0.55,
                medianprops={"color": "#1f1f1f", "linewidth": 1.5},
                boxprops={"linewidth": 1.1},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
            )
            for patch, method in zip(box["boxes"], METHODS):
                patch.set_facecolor(COLORS[method])
                patch.set_alpha(0.80)
            if row_idx == 0:
                ax.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")
            else:
                ax.set_title("")
            ax.set_xticks(range(1, len(METHODS) + 1))
            ax.set_xticklabels(METHODS, rotation=18, ha="right", fontsize=9)
            ax.set_ylim(*YLIMS[metric])
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
            ax.set_axisbelow(True)

    axes[0][0].set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    axes[1][0].set_ylabel("Recall", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hash_accuracy_recall_boxplots.png", dpi=600)
    fig.savefig(FIG_DIR / "hash_accuracy_recall_boxplots.pdf")
    plt.close(fig)


def write_readme() -> None:
    text = """# Reference hash boxplot sample

This directory contains a constructed reference sample, not experimental data.
It is intended only as a plotting and layout example for comparing MECH,
HyperplaneLSH, SimHash, and AngularLSH across four datasets and several
hash-code lengths.

Files:

- `reference_hash_metrics.csv`: per-dataset, per-method, per-hash-bit values.
- `reference_hash_summary.csv`: mean, quartiles, IQR, and range for each box.
- `figures/hash_accuracy_boxplot.png`: accuracy boxplots.
- `figures/hash_recall_boxplot.png`: recall boxplots.
- `figures/hash_accuracy_recall_boxplots.png`: combined 2 x 4 accuracy and recall view.

The `source_note` column deliberately marks each row as
`reference_constructed_randomized_non_experimental` so the values are not
mistaken for measurements from an actual run.
"""
    (OUT_DIR / "README.md").write_text(text)


def main() -> None:
    rows = build_rows()
    write_rows(rows)
    write_summary(rows)
    draw_metric_boxplot(rows, "accuracy")
    draw_metric_boxplot(rows, "recall")
    draw_combined_boxplot(rows)
    write_readme()


if __name__ == "__main__":
    main()
