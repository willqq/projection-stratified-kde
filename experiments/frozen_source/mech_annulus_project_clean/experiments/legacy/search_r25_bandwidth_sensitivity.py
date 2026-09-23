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
OUT_DIR = Path("r25_bandwidth_sensitivity")
RADIUS_PERCENTILE = 25.0
BANDWIDTH_FACTORS = [round(value / 100.0, 2) for value in range(1, 26)]
EPS = exp.EPS


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, dict]:
    exp.set_seed(exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    x = load_reference_dataset(dataset)
    sizes = split_sizes(len(x))
    _, index_x, queries = exp.split_data(
        x,
        sizes["n_train"],
        sizes["n_index"],
        sizes["n_queries"],
        rng,
    )
    distances = exp.distance_matrix(index_x, queries)
    query_r25 = np.percentile(distances, RADIUS_PERCENTILE, axis=1)
    fixed_r25 = float(np.median(query_r25))
    fixed_mask = distances <= fixed_r25
    adaptive_mask = distances <= query_r25[:, None]

    rows = []
    for bandwidth_factor in BANDWIDTH_FACTORS:
        bandwidth = fixed_r25 * float(bandwidth_factor)
        weights = gaussian_kernel(distances, bandwidth)
        global_kde = weights.sum(axis=1)
        nonzero_global = global_kde > EPS

        fixed_local_kde = (weights * fixed_mask).sum(axis=1)
        adaptive_local_kde = (weights * adaptive_mask).sum(axis=1)
        fixed_error = np.abs(fixed_local_kde - global_kde) / np.maximum(global_kde, EPS)
        adaptive_error = np.abs(adaptive_local_kde - global_kde) / np.maximum(global_kde, EPS)

        rows.append(
            {
                "dataset": dataset,
                "bandwidth_factor": float(bandwidth_factor),
                "bandwidth": float(bandwidth),
                "fixed_r25": fixed_r25,
                "mean_fixed_r25_error": float(np.mean(fixed_error)),
                "median_fixed_r25_error": float(np.median(fixed_error)),
                "p90_fixed_r25_error": float(np.percentile(fixed_error, 90)),
                "max_fixed_r25_error": float(np.max(fixed_error)),
                "mean_adaptive_r25_error": float(np.mean(adaptive_error)),
                "median_adaptive_r25_error": float(np.median(adaptive_error)),
                "p90_adaptive_r25_error": float(np.percentile(adaptive_error, 90)),
                "max_adaptive_r25_error": float(np.max(adaptive_error)),
                "mean_fixed_r25_coverage": float(
                    np.mean(fixed_local_kde / np.maximum(global_kde, EPS))
                ),
                "mean_adaptive_r25_coverage": float(
                    np.mean(adaptive_local_kde / np.maximum(global_kde, EPS))
                ),
                "mean_global_kde": float(np.mean(global_kde)),
                "min_global_kde": float(np.min(global_kde)),
                "nonzero_global_kde_fraction": float(np.mean(nonzero_global)),
                "mean_fixed_candidate_size": float(np.mean(fixed_mask.sum(axis=1))),
                "mean_adaptive_candidate_size": float(np.mean(adaptive_mask.sum(axis=1))),
            }
        )

    meta = {
        "dataset": dataset,
        "preprocessing": "reference_minmax_without_origin_shift",
        "split": sizes,
        "n_index": int(len(index_x)),
        "n_queries": int(len(queries)),
        "radius_percentile": RADIUS_PERCENTILE,
        "fixed_r25": fixed_r25,
        "factor_definition": "bandwidth = fixed_r25 * bandwidth_factor",
        "bandwidth_factors": BANDWIDTH_FACTORS,
    }
    return pd.DataFrame(rows), meta


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.pivot_table(
            index="bandwidth_factor",
            columns="dataset",
            values=metric,
            aggfunc="mean",
        )
        .reindex(index=BANDWIDTH_FACTORS, columns=DATASETS)
        .reset_index()
    )


def summarize_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        part = part.sort_values("bandwidth_factor")
        reliable = part[part["nonzero_global_kde_fraction"] >= 0.95]
        for threshold in [0.10, 0.05, 0.02, 0.01]:
            candidates = reliable[reliable["mean_fixed_r25_error"] <= threshold]
            if candidates.empty:
                selected = reliable.iloc[0] if not reliable.empty else part.iloc[-1]
                met = False
            else:
                selected = candidates.iloc[0]
                met = True
            rows.append(
                {
                    "dataset": dataset,
                    "error_threshold": threshold,
                    "target_met": bool(met),
                    "recommended_bandwidth_factor": float(selected["bandwidth_factor"]),
                    "mean_fixed_r25_error": float(selected["mean_fixed_r25_error"]),
                    "mean_adaptive_r25_error": float(selected["mean_adaptive_r25_error"]),
                    "nonzero_global_kde_fraction": float(
                        selected["nonzero_global_kde_fraction"]
                    ),
                    "mean_global_kde": float(selected["mean_global_kde"]),
                }
            )
    return pd.DataFrame(rows)


def plot_curves(df: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 22,
            "font.weight": "normal",
            "axes.titlesize": 24,
            "axes.titleweight": "normal",
            "axes.labelsize": 24,
            "axes.labelweight": "normal",
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 16,
            "savefig.dpi": 600,
        }
    )
    colors = {
        "isolet": "#4c78a8",
        "cifar10": "#59a14f",
        "cifar10_gist512": "#f28e2b",
        "amazon": "#e15759",
    }
    labels = {
        "isolet": "ISOLET",
        "cifar10": "CIFAR-10",
        "cifar10_gist512": "CIFAR10-GIST512",
        "amazon": "Amazon",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 7.2))
    for dataset, part in df.groupby("dataset"):
        part = part.sort_values("bandwidth_factor")
        axes[0].plot(
            part["bandwidth_factor"],
            part["mean_fixed_r25_error"],
            linestyle="-",
            marker="o",
            markersize=6.4,
            linewidth=2.2,
            color=colors.get(dataset),
            label=labels.get(dataset, dataset),
        )
        axes[1].plot(
            part["bandwidth_factor"],
            part["nonzero_global_kde_fraction"],
            linestyle="-",
            marker="o",
            markersize=6.4,
            linewidth=2.2,
            color=colors.get(dataset),
            label=labels.get(dataset, dataset),
        )

    axes[0].set_title("(a) Truncation error", pad=14)
    axes[0].set_ylabel("Mean abs. relative error")
    axes[1].set_title("(b) Numerical nonzero global KDE fraction", pad=14)
    axes[1].set_ylabel("Fraction")

    for ax in axes:
        ax.set_xlabel("Bandwidth factor")
        ax.set_xticks(BANDWIDTH_FACTORS[::2])
        ax.tick_params(axis="both", which="major", width=1.8, length=8)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("normal")
            tick_label.set_fontfamily("Times New Roman")
        ax.grid(True, alpha=0.28, linewidth=1.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
    legend = axes[0].legend(
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.28,
        borderaxespad=0.25,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")
        text.set_fontfamily("Times New Roman")
    fig.tight_layout()
    fig.savefig(figure_dir / "r25_bandwidth_sensitivity_curves.png", dpi=600)
    fig.savefig(figure_dir / "r25_bandwidth_sensitivity_curves.pdf")
    plt.close(fig)


def plot_table(table: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    display = table.copy()
    for col in display.columns:
        if col == "bandwidth_factor":
            display[col] = display[col].map(lambda value: f"{float(value):.2f}")
        else:
            display[col] = display[col].map(lambda value: f"{float(value):.4f}")

    fig, ax = plt.subplots(figsize=(8.6, 11.0))
    ax.axis("off")
    table_artist = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8.0)
    table_artist.scale(1.0, 1.18)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#fafafa")
            cell.set_text_props(weight="bold")
    ax.set_title("Mean fixed R25 KDE truncation error", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "r25_bandwidth_error_table.png", dpi=240)
    fig.savefig(figure_dir / "r25_bandwidth_error_table.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metadata = []
    for dataset in DATASETS:
        rows, meta = evaluate_dataset(dataset)
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows.to_csv(dataset_dir / "r25_bandwidth_sensitivity.csv", index=False)
        all_rows.append(rows)
        metadata.append(meta)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT_DIR / "r25_bandwidth_sensitivity_all.csv", index=False)

    fixed_error_table = pivot_metric(df, "mean_fixed_r25_error")
    adaptive_error_table = pivot_metric(df, "mean_adaptive_r25_error")
    nonzero_table = pivot_metric(df, "nonzero_global_kde_fraction")
    fixed_error_table.to_csv(OUT_DIR / "fixed_r25_error_table.csv", index=False)
    adaptive_error_table.to_csv(OUT_DIR / "adaptive_r25_error_table.csv", index=False)
    nonzero_table.to_csv(OUT_DIR / "global_kde_nonzero_fraction_table.csv", index=False)

    recommendations = summarize_thresholds(df)
    recommendations.to_csv(OUT_DIR / "fixed_r25_bandwidth_recommendations.csv", index=False)

    with open(OUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": DATASETS,
                "radius_percentile": RADIUS_PERCENTILE,
                "bandwidth_factors": BANDWIDTH_FACTORS,
                "metric": "mean(abs(KDE_R25 - KDE_global) / max(KDE_global, EPS))",
                "reliability_filter": "nonzero_global_kde_fraction >= 0.95",
                "metadata": metadata,
            },
            f,
            indent=2,
        )

    plot_curves(df)
    plot_table(fixed_error_table)
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
