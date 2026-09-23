from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech_annulus_experiments as exp
from run_multidataset_requested_sensitivity import split_sizes
from run_r25_lk_sensitivity import (
    BANDWIDTH_FACTOR,
    BASE,
    DATASETS,
    K_VALUES,
    L_VALUES,
    RADIUS_PERCENTILE,
    evaluate_adaptive_r25,
    query_adaptive_r25,
)


OUT_DIR = Path("r25_lk_grid_sensitivity")
DELTA_FACTOR = 0.25


def score_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kde_quality"] = (
        1.0 - df["kde_abs_relative_error_vs_exact_r25"]
    ).clip(lower=0.0, upper=1.0)
    scored = []
    for dataset, part in df.groupby("dataset"):
        part = part.copy()
        time_max = max(float(part["query_time_ms"].max()), 1e-12)
        size_max = max(float(part["candidate_size"].max()), 1e-12)
        part["speed_quality"] = (1.0 - part["query_time_ms"] / time_max).clip(0.0, 1.0)
        part["size_quality"] = (1.0 - part["candidate_size"] / size_max).clip(0.0, 1.0)
        part["score_accuracy"] = (
            0.45 * part["point_recall"]
            + 0.35 * part["kde_quality"]
            + 0.20 * part["point_f1"]
        )
        part["score_balanced"] = (
            0.35 * part["point_recall"]
            + 0.30 * part["kde_quality"]
            + 0.20 * part["point_f1"]
            + 0.10 * part["speed_quality"]
            + 0.05 * part["size_quality"]
        )
        part["feasible"] = (
            (part["point_recall"] >= 0.90)
            & (part["kde_abs_relative_error_vs_exact_r25"] <= 0.10)
        )
        scored.append(part)
    return pd.concat(scored, ignore_index=True)


def best_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        feasible = part[part["feasible"]]
        source = feasible if not feasible.empty else part
        best_balanced = source.sort_values(
            ["score_balanced", "point_recall", "kde_quality", "query_time_ms"],
            ascending=[False, False, False, True],
        ).iloc[0].copy()
        best_balanced["selection"] = "balanced"
        rows.append(best_balanced)

        best_accuracy = source.sort_values(
            ["score_accuracy", "point_recall", "kde_quality", "query_time_ms"],
            ascending=[False, False, False, True],
        ).iloc[0].copy()
        best_accuracy["selection"] = "accuracy"
        rows.append(best_accuracy)
    return pd.DataFrame(rows)


def evaluate_dataset(dataset: str) -> tuple[pd.DataFrame, dict]:
    exp.set_seed(exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    x = exp.load_dataset(dataset)
    sizes = split_sizes(len(x))
    train_x, index_x, queries = exp.split_data(
        x,
        sizes["n_train"],
        sizes["n_index"],
        sizes["n_queries"],
        rng,
    )
    distances = exp.distance_matrix(index_x, queries)
    query_radii = query_adaptive_r25(distances)
    median_r25 = float(np.median(query_radii))
    bandwidths = query_radii * BANDWIDTH_FACTOR
    delta = max(median_r25 * DELTA_FACTOR, exp.EPS)

    model = exp.train_mech(
        train_x,
        max(L_VALUES),
        max(K_VALUES),
        24,
        128,
        1e-3,
        1.0,
        0.1,
        0.1,
        0.1,
        "cpu",
    )

    rows = []
    total = len(L_VALUES) * len(K_VALUES)
    done = 0
    for L in L_VALUES:
        for K in K_VALUES:
            index = exp.MECHHashIndex(
                model,
                index_x,
                int(L),
                int(K),
                int(BASE["hamming_probe"]),
            )
            row = evaluate_adaptive_r25(
                index,
                index_x,
                queries,
                distances,
                query_radii,
                bandwidths,
                delta,
                int(BASE["min_collisions"]),
                int(BASE["annulus_count"]),
            )
            row.update(
                {
                    "dataset": dataset,
                    "L": int(L),
                    "K": int(K),
                    "hamming_probe": int(BASE["hamming_probe"]),
                    "median_r25": median_r25,
                    "mean_r25": float(np.mean(query_radii)),
                    "mean_bandwidth": float(np.mean(bandwidths)),
                }
            )
            rows.append(row)
            done += 1
            print(
                f"[{dataset} {done:03d}/{total}] L={L} K={K}: "
                f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                f"F1={row['point_f1']:.3f} "
                f"KDE={row['kde_abs_relative_error_vs_exact_r25']:.3f} "
                f"C={row['candidate_size']:.1f} T={row['query_time_ms']:.3f}ms",
                flush=True,
            )

    meta = {
        "dataset": dataset,
        "split": sizes,
        "n_train": int(len(train_x)),
        "n_index": int(len(index_x)),
        "n_queries": int(len(queries)),
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "median_r25": median_r25,
        "mean_r25": float(np.mean(query_radii)),
        "bandwidth_mode": "h(q)=0.14*R25(q)",
        "mean_bandwidth": float(np.mean(bandwidths)),
        "delta": delta,
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "base": BASE,
    }
    return pd.DataFrame(rows), meta


def plot_heatmap(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    title: str,
    filename: str,
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
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    image = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(L_VALUES)))
    ax.set_xticklabels(L_VALUES)
    ax.set_yticks(range(len(K_VALUES)))
    ax.set_yticklabels(K_VALUES)
    ax.set_xlabel("Hash tables L")
    ax.set_ylabel("Hash code length K")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{filename}.png", dpi=450)
    fig.savefig(figure_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_all_heatmaps(df: pd.DataFrame) -> None:
    specs = [
        ("point_recall", "Recall", "recall_heatmap"),
        ("kde_abs_relative_error_vs_exact_r25", "KDE error", "kde_error_heatmap"),
        ("candidate_size", "Candidate size", "candidate_size_heatmap"),
        ("query_time_ms", "Query time (ms)", "query_time_heatmap"),
        ("score_balanced", "Balanced score", "balanced_score_heatmap"),
    ]
    for dataset in DATASETS:
        for metric, title, filename in specs:
            plot_heatmap(df, dataset, metric, f"{dataset}: {title}", filename)


def pivot_best_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.pivot_table(index=["dataset", "K"], columns="L", values=metric, aggfunc="mean")
        .reset_index()
    )


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
            elif col in {"L", "K"}:
                cells.append(str(int(value)))
            elif col == "feasible":
                cells.append(str(bool(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metadata = []
    for dataset in DATASETS:
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result_path = dataset_dir / "r25_lk_grid.csv"
        if result_path.exists():
            print(f"Using existing grid results for {dataset}", flush=True)
            df = pd.read_csv(result_path)
            meta = {"dataset": dataset, "loaded_existing": True}
        else:
            print(f"=== Dataset: {dataset} ===", flush=True)
            df, meta = evaluate_dataset(dataset)
            df.to_csv(result_path, index=False)
        all_rows.append(df)
        metadata.append(meta)

    results = score_grid(pd.concat(all_rows, ignore_index=True))
    results.to_csv(OUT_DIR / "r25_lk_grid_all.csv", index=False)
    best = best_by_dataset(results)
    best.to_csv(OUT_DIR / "best_lk_by_dataset.csv", index=False)

    for metric in [
        "point_recall",
        "point_precision",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "query_time_ms",
        "score_balanced",
    ]:
        pivot_best_metric(results, metric).to_csv(OUT_DIR / f"grid_{metric}.csv", index=False)

    plot_all_heatmaps(results)

    config = {
        "datasets": DATASETS,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "bandwidth_mode": "h(q)=0.14*R25(q)",
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta_mode": "delta=0.25*median_q R25(q)",
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "base": BASE,
        "selection": {
            "feasible": "recall >= 0.90 and KDE error <= 0.10",
            "score_accuracy": "0.45*recall + 0.35*kde_quality + 0.20*F1",
            "score_balanced": "0.35*recall + 0.30*kde_quality + 0.20*F1 + 0.10*speed_quality + 0.05*size_quality",
        },
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_cols = [
        "dataset",
        "selection",
        "L",
        "K",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "query_time_ms",
        "score_balanced",
        "score_accuracy",
        "feasible",
    ]
    summary = [
        "# R25 L/K Grid Sensitivity",
        "",
        "固定查询半径为 query-adaptive R25，核带宽为 h(q)=0.14R25(q)，对哈希表数量 L 和哈希码长度 K 做双因子网格敏感性分析。",
        "",
        "## Best L/K by Dataset",
        "",
        markdown_table(best, best_cols),
        "",
        "## Outputs",
        "",
        "- `r25_lk_grid_all.csv`",
        "- `best_lk_by_dataset.csv`",
        "- `figures/<dataset>/recall_heatmap.png`",
        "- `figures/<dataset>/kde_error_heatmap.png`",
        "- `figures/<dataset>/balanced_score_heatmap.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved R25 L/K grid sensitivity results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
