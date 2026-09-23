from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("r25_bandwidth_sensitivity/figures")
TABLE_PATH = Path("r25_bandwidth_sensitivity/adaptive_r25_error_table.csv")
PNG_PATH = OUT_DIR / "adaptive_r25_truncation_error_highres.png"
PDF_PATH = OUT_DIR / "adaptive_r25_truncation_error_highres.pdf"


COLORS = {
    "isolet": "#4c78a8",
    "cifar10": "#59a14f",
    "cifar10_gist512": "#f28e2b",
    "amazon": "#e15759",
}

LABELS = {
    "isolet": "ISOLET",
    "cifar10": "CIFAR-10",
    "cifar10_gist512": "CIFAR10-GIST512",
    "amazon": "Amazon",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE_PATH)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 17,
            "font.weight": "bold",
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "lines.linewidth": 3.0,
            "savefig.dpi": 600,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    x = table["bandwidth_factor"]
    for dataset in ["isolet", "cifar10", "cifar10_gist512", "amazon"]:
        ax.plot(
            x,
            table[dataset],
            marker="o",
            markersize=7.0,
            linewidth=3.0,
            color=COLORS[dataset],
            label=LABELS[dataset],
        )

    ax.axvline(0.14, color="#333333", linestyle="--", linewidth=2.0)
    ax.axhline(0.03, color="#777777", linestyle=":", linewidth=2.0)
    ax.text(
        0.145,
        0.315,
        "selected factor = 0.14",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        0.247,
        0.033,
        "3% error",
        ha="right",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color="#555555",
    )

    ax.set_title("KDE Truncation Error under Query-Adaptive R25", pad=16)
    ax.set_xlabel("Bandwidth factor", labelpad=10)
    ax.set_ylabel("Mean truncation error", labelpad=10)
    ax.set_xlim(0.01, 0.25)
    ax.set_ylim(0.0, 0.36)
    ax.set_xticks([0.01, 0.05, 0.10, 0.14, 0.15, 0.20, 0.25])
    ax.tick_params(axis="both", which="major", width=1.4, length=6)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontfamily("Times New Roman")
    ax.grid(True, alpha=0.28, linewidth=1.1)
    legend = ax.legend(frameon=False, loc="upper left", handlelength=2.5)
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_fontfamily("Times New Roman")
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    fig.tight_layout()

    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
