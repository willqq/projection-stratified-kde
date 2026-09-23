from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "navy": "#263A63",
    "blue": "#2F6B9A",
    "teal": "#2F8F83",
    "green": "#77A65B",
    "gold": "#D9A441",
    "orange": "#C96F3D",
    "red": "#B55252",
    "purple": "#7A5C9E",
    "gray": "#6F7682",
    "light": "#F4F6F8",
    "line": "#263238",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#B8C0CC",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D8DEE8",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#1F2937",
            "axes.titleweight": "bold",
        }
    )


def save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pct(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def annotate_bars(ax: plt.Axes, bars, fmt="{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )


def plot_hash_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.18
    highlight_idx = methods.index("MECH") if "MECH" in methods else None

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    fig.suptitle("Hash Method Comparison on ISOLET", x=0.02, y=1.03, ha="left")

    ax = axes[0]
    if highlight_idx is not None:
        ax.axvspan(highlight_idx - 0.48, highlight_idx + 0.48, color="#EAF4EF", zorder=0)
    metric_defs = [
        ("Precision", "point_precision", PALETTE["blue"]),
        ("Recall", "point_recall", PALETTE["teal"]),
        ("F1", "point_f1", PALETTE["gold"]),
        (r"$R^k$", "kernel_weighted_recall", PALETTE["purple"]),
    ]
    for i, (label, col, color) in enumerate(metric_defs):
        bars = ax.bar(x + (i - 1.5) * width, pct(df[col]), width, label=label, color=color, zorder=3)
        if col == "point_f1":
            annotate_bars(ax, bars)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 122)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_title("Point-Level and Kernel-Weighted Recall")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))

    ax = axes[1]
    if highlight_idx is not None:
        ax.axvspan(highlight_idx - 0.48, highlight_idx + 0.48, color="#EAF4EF", zorder=0)
    bars = ax.bar(
        x - width / 2,
        pct(df["kde_abs_relative_error"]),
        width,
        label="KDE Error",
        color=PALETTE["red"],
        zorder=3,
    )
    annotate_bars(ax, bars)
    ax2 = ax.twinx()
    ax2.plot(
        x + width / 2,
        df["candidate_expansion_ratio"],
        marker="o",
        linewidth=2.4,
        label="CER",
        color=PALETTE["navy"],
    )
    ax.set_ylabel("KDE Error (%)")
    ax2.set_ylabel("Candidate Expansion Ratio")
    ax.set_title("KDE Error and Candidate Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylim(0, max(10, pct(df["kde_abs_relative_error"]).max() * 1.35))
    ax2.set_ylim(0, max(1.2, df["candidate_expansion_ratio"].max() * 1.25))
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    save(fig, out_dir / "mech_hash_method_comparison")


def plot_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    df = df[~df["variant"].str.contains("Oracle", case=False, na=False)].copy()
    variants = df["variant"].tolist()
    x = np.arange(len(variants))
    full_idx = variants.index("Full MECH") if "Full MECH" in variants else None

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))
    fig.suptitle("MECH Structure Ablation on ISOLET", x=0.02, y=1.03, ha="left")

    ax = axes[0]
    if full_idx is not None:
        ax.axvspan(full_idx - 0.48, full_idx + 0.48, color="#EAF4EF", zorder=0)
    width = 0.34
    recall_colors = [PALETTE["teal"] if v != "Full MECH" else PALETTE["green"] for v in variants]
    weighted_colors = [PALETTE["purple"] if v != "Full MECH" else PALETTE["navy"] for v in variants]
    edge_colors = [PALETTE["line"] if v == "Full MECH" else "white" for v in variants]
    bars1 = ax.bar(
        x - width / 2,
        pct(df["point_recall"]),
        width,
        label="Recall",
        color=recall_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        zorder=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        pct(df["kernel_weighted_recall"]),
        width,
        label=r"$R^k$",
        color=weighted_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        zorder=3,
    )
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=18, ha="right")
    ax.set_title("Annulus Recall Quality")
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, loc="upper left")
    annotate_bars(ax, bars2)
    if full_idx is not None:
        full_rk = df.iloc[full_idx]["kernel_weighted_recall"] * 100.0
        ax.annotate(
            "Full MECH",
            xy=(full_idx + width / 2, full_rk),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["navy"],
            arrowprops={"arrowstyle": "->", "color": PALETTE["navy"], "lw": 1.1},
        )

    ax = axes[1]
    if full_idx is not None:
        ax.axvspan(full_idx - 0.48, full_idx + 0.48, color="#EAF4EF", zorder=0)
    fp_values = pct(df["kernel_weighted_fp_ratio"])
    fp_colors = [PALETTE["orange"] if v != "Full MECH" else PALETTE["green"] for v in variants]
    bars = ax.bar(
        x,
        fp_values,
        color=fp_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        zorder=3,
    )
    for bar, value in zip(bars, fp_values):
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, max(float(value), 0.8)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(0, max(1000, fp_values.max() * 1.25))
    ax.set_ylabel(r"$E_+^k$ (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=18, ha="right")
    ax.set_title("Weighted False Positives")

    ax = axes[2]
    if full_idx is not None:
        ax.axvspan(full_idx - 0.48, full_idx + 0.48, color="#EAF4EF", zorder=0)
    err_colors = [PALETTE["red"] if v != "Full MECH" else PALETTE["green"] for v in variants]
    err_values = pct(df["kde_abs_relative_error"])
    bars = ax.bar(
        x,
        err_values,
        color=err_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        label="KDE Error",
        zorder=3,
    )
    for bar, value in zip(bars, err_values):
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, max(float(value), 0.8)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        df["candidate_expansion_ratio"],
        marker="D",
        linewidth=2.4,
        color=PALETTE["navy"],
        label="CER",
    )
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("KDE Error (%)")
    ax2.set_ylabel("CER")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=18, ha="right")
    ax.set_title("KDE Error and Candidate Cost")
    ax.set_ylim(0, max(1000, err_values.max() * 1.25))
    ax2.set_ylim(0, max(1.2, df["candidate_expansion_ratio"].max() * 1.20))
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    save(fig, out_dir / "mech_structure_ablation")


def plot_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    groups = ["L", "K", "delta", "rho", "sigma"]
    titles = {
        "L": "Hash Tables",
        "K": "Hash Code Length",
        "delta": "Sphere Interval",
        "rho": "Annulus Radius Factor",
        "sigma": "Bandwidth Factor",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2))
    axes = axes.ravel()
    fig.suptitle("MECH Parameter Sensitivity", x=0.02, y=1.02, ha="left")

    for ax, group in zip(axes, groups):
        sub = df[df["parameter"] == group].sort_values("value")
        x = sub["value"].astype(float).to_numpy()
        ax.plot(x, pct(sub["kernel_weighted_recall"]), marker="o", linewidth=2.2, color=PALETTE["purple"], label=r"$R^k$")
        ax.plot(x, pct(sub["kde_abs_relative_error"]), marker="s", linewidth=2.2, color=PALETTE["red"], label="KDE Error")
        ax.set_title(titles[group])
        ax.set_xlabel(group)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, max(100, pct(sub[["kernel_weighted_recall", "kde_abs_relative_error"]]).max().max() * 1.12))
        ax2 = ax.twinx()
        ax2.plot(x, sub["candidate_expansion_ratio"], marker="D", linestyle="--", linewidth=1.9, color=PALETTE["navy"], label="CER")
        ax2.set_ylabel("CER")
        ax2.set_ylim(0, max(1.4, sub["candidate_expansion_ratio"].max() * 1.25))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    axes[-1].axis("off")
    axes[-1].text(
        0.0,
        0.92,
        "Reading the curves",
        fontsize=13,
        fontweight="bold",
        transform=axes[-1].transAxes,
    )
    axes[-1].text(
        0.0,
        0.74,
        "Higher $R^k$ means the retrieved annulus preserves\nmore KDE contribution. Lower KDE error and CER\nindicate a better accuracy-cost balance.",
        fontsize=10.5,
        linespacing=1.5,
        transform=axes[-1].transAxes,
    )
    axes[-1].text(
        0.0,
        0.36,
        "In this run, smaller code length and finer sphere\ninterval improve weighted recall, while increasing\nL improves KDE error at higher online cost.",
        fontsize=10.5,
        linespacing=1.5,
        transform=axes[-1].transAxes,
    )

    fig.tight_layout()
    save(fig, out_dir / "mech_parameter_sensitivity")


def plot_error_propagation(method_df: pd.DataFrame, sens_df: pd.DataFrame, out_dir: Path) -> None:
    method_points = method_df.copy()
    method_points["label"] = method_points["method"]
    sens_points = sens_df.copy()
    sens_points["label"] = sens_points["parameter"] + "=" + sens_points["value"].astype(str)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.scatter(
        pct(sens_points["kernel_weighted_recall"]),
        pct(sens_points["kde_abs_relative_error"]),
        s=70,
        alpha=0.78,
        color=PALETTE["teal"],
        edgecolor="white",
        linewidth=0.8,
        label="MECH sensitivity",
    )
    ax.scatter(
        pct(method_points["kernel_weighted_recall"]),
        pct(method_points["kde_abs_relative_error"]),
        s=115,
        marker="D",
        color=PALETTE["gold"],
        edgecolor=PALETTE["line"],
        linewidth=0.9,
        label="Hash methods",
    )
    for _, row in method_points.iterrows():
        if row["label"] != "MECH":
            continue
        ax.annotate(
            "MECH / full pipeline",
            (row["kernel_weighted_recall"] * 100, row["kde_abs_relative_error"] * 100),
            xytext=(-96, 18),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["navy"],
            arrowprops={"arrowstyle": "->", "color": PALETTE["navy"], "lw": 1.0},
        )
    for _, row in sens_points.iterrows():
        label = row["label"]
        if label in {"delta=0.5", "delta=1.0", "delta=2.0", "rho=0.75"}:
            ax.annotate(
                label.replace(".0", ""),
                (row["kernel_weighted_recall"] * 100, row["kde_abs_relative_error"] * 100),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                color="#334155",
            )
    ax.set_xlabel(r"Kernel-Weighted Recall $R^k$ (%)")
    ax.set_ylabel("KDE Absolute Relative Error (%)")
    ax.set_title("Error Propagation: Weighted Recall vs KDE Error")
    ax.legend(loc="upper right")
    all_x = pd.concat([pct(sens_points["kernel_weighted_recall"]), pct(method_points["kernel_weighted_recall"])])
    all_y = pd.concat([pct(sens_points["kde_abs_relative_error"]), pct(method_points["kde_abs_relative_error"])])
    ax.set_xlim(max(0, all_x.min() - 3), min(101, all_x.max() + 2))
    ax.set_ylim(0, max(6, all_y.max() * 1.35))
    fig.tight_layout()
    save(fig, out_dir / "mech_error_propagation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="mech_experiment_results")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "figures"
    setup_style()

    method_df = pd.read_csv(result_dir / "hash_method_comparison.csv")
    ablation_df = pd.read_csv(result_dir / "structure_ablation.csv")
    sensitivity_df = pd.read_csv(result_dir / "mech_sensitivity_summary.csv")

    plot_hash_comparison(method_df, out_dir)
    plot_ablation(ablation_df, out_dir)
    plot_sensitivity(sensitivity_df, out_dir)
    plot_error_propagation(method_df, sensitivity_df, out_dir)

    print(f"Saved figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
