from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import mech_annulus_experiments as exp
from run_multidataset_requested_sensitivity import split_sizes


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("r25_k10_200_l20_sensitivity")

L_FIXED = 20
K_VALUES = list(range(10, 201, 10))
MAX_K = max(K_VALUES)
RADIUS_PERCENTILE = 25.0
BANDWIDTH_FACTOR = 0.14
DELTA_FACTOR = 0.25
MIN_COLLISIONS = 2
HAMMING_PROBE = 2


def query_adaptive_r25(distances: np.ndarray) -> np.ndarray:
    return np.percentile(distances, RADIUS_PERCENTILE, axis=1).astype(np.float32)


def encode_tables(
    model: exp.MultiEncoderContrastiveHash,
    x: np.ndarray,
) -> list[np.ndarray]:
    with torch.no_grad():
        encoded = model.encode_continuous(torch.from_numpy(x.astype(np.float32)))
    return [h.numpy().astype(np.float32) for h in encoded]


def sphere_candidates_array(
    sphere_table: dict[int, np.ndarray],
    q_norm: float,
    radius: float,
    delta: float,
) -> np.ndarray:
    low, high = exp.sphere_layer_bounds(q_norm, radius, delta)
    rows = [sphere_table[layer] for layer in range(low, high + 1) if layer in sphere_table]
    if not rows:
        return np.asarray([], dtype=np.int32)
    return np.unique(np.concatenate(rows).astype(np.int32))


def evaluate_k(
    dataset: str,
    k_value: int,
    index_bits: list[np.ndarray],
    query_bits: list[np.ndarray],
    index_x: np.ndarray,
    queries: np.ndarray,
    distances: np.ndarray,
    query_radii: np.ndarray,
    bandwidths: np.ndarray,
    delta: float,
    build_time_s: float,
) -> dict[str, float | int | str]:
    norms = np.linalg.norm(index_x, axis=1)
    sphere_table = exp.build_sphere_table(norms, delta)
    query_norms = np.linalg.norm(queries, axis=1)

    precision_values = []
    recall_values = []
    f1_values = []
    candidate_sizes = []
    exact_r25_sizes = []
    kde_errors = []
    global_kde_errors = []
    weighted_recalls = []
    hamming_radii = []
    query_times = []

    for q_id, q in enumerate(queries):
        radius = float(query_radii[q_id])
        bandwidth = float(bandwidths[q_id])
        d = distances[q_id]
        true_ids = np.flatnonzero(d <= radius).astype(np.int32)
        if len(true_ids) == 0:
            continue

        weights = exp.gaussian_kernel(d, bandwidth)
        true_kde = float(weights[true_ids].sum())
        global_kde = float(weights.sum())

        start = time.perf_counter()
        candidate_pool = sphere_candidates_array(
            sphere_table,
            float(query_norms[q_id]),
            radius,
            delta,
        )
        if len(candidate_pool) == 0:
            approx_ids = np.asarray([], dtype=np.int32)
            hamming_radius = HAMMING_PROBE
        else:
            angle = exp.max_angle_from_radius(float(query_norms[q_id]), radius)
            hamming_radius = max(
                HAMMING_PROBE,
                exp.hamming_radius_from_angle(angle, k_value),
            )
            counts = np.zeros(len(candidate_pool), dtype=np.int16)
            for table_id in range(L_FIXED):
                distances_h = np.count_nonzero(
                    index_bits[table_id][candidate_pool, :k_value]
                    != query_bits[table_id][q_id, :k_value],
                    axis=1,
                )
                counts += (distances_h <= hamming_radius).astype(np.int16)
            approx_ids = candidate_pool[counts >= MIN_COLLISIONS]
        if len(approx_ids) > 0:
            approx_kde = float(weights[approx_ids].sum())
        else:
            approx_kde = 0.0
        query_times.append(time.perf_counter() - start)
        hamming_radii.append(hamming_radius)

        true_set = set(true_ids.tolist())
        approx_set = set(approx_ids.tolist())
        tp = np.asarray(list(true_set & approx_set), dtype=np.int32)

        precision = len(tp) / len(approx_ids) if len(approx_ids) else 0.0
        recall = len(tp) / len(true_ids)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(2.0 * precision * recall / (precision + recall + exp.EPS))
        candidate_sizes.append(len(approx_ids))
        exact_r25_sizes.append(len(true_ids))
        kde_errors.append(abs(approx_kde - true_kde) / max(true_kde, exp.EPS))
        global_kde_errors.append(abs(approx_kde - global_kde) / max(global_kde, exp.EPS))
        tp_weight = float(weights[tp].sum()) if len(tp) else 0.0
        weighted_recalls.append(tp_weight / max(true_kde, exp.EPS))

    return {
        "dataset": dataset,
        "L": L_FIXED,
        "K": int(k_value),
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta": float(delta),
        "min_collisions": MIN_COLLISIONS,
        "hamming_probe": HAMMING_PROBE,
        "point_precision": float(np.mean(precision_values)),
        "point_recall": float(np.mean(recall_values)),
        "point_f1": float(np.mean(f1_values)),
        "kernel_weighted_recall": float(np.mean(weighted_recalls)),
        "kde_abs_relative_error_vs_exact_r25": float(np.mean(kde_errors)),
        "kde_abs_relative_error_vs_global": float(np.mean(global_kde_errors)),
        "candidate_size": float(np.mean(candidate_sizes)),
        "exact_r25_candidate_size": float(np.mean(exact_r25_sizes)),
        "candidate_expansion_ratio": float(
            np.mean(np.asarray(candidate_sizes) / np.maximum(exact_r25_sizes, 1))
        ),
        "mean_hamming_radius": float(np.mean(hamming_radii)),
        "query_time_ms": float(np.mean(query_times) * 1000.0),
        "build_time_s": float(build_time_s),
    }


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

    train_start = time.perf_counter()
    model = exp.train_mech(
        train_x,
        L_FIXED,
        MAX_K,
        24,
        128,
        1e-3,
        1.0,
        0.1,
        0.1,
        0.1,
        "cpu",
    )
    train_time_s = time.perf_counter() - train_start

    encode_start = time.perf_counter()
    index_encoded = encode_tables(model, index_x)
    query_encoded = encode_tables(model, queries)
    thresholds = [np.median(h, axis=0) for h in index_encoded]
    index_bits = [h >= thresholds[i] for i, h in enumerate(index_encoded)]
    query_bits = [h >= thresholds[i] for i, h in enumerate(query_encoded)]
    encode_time_s = time.perf_counter() - encode_start
    build_time_s = train_time_s + encode_time_s

    rows = []
    for idx, k_value in enumerate(K_VALUES, start=1):
        row = evaluate_k(
            dataset,
            k_value,
            index_bits,
            query_bits,
            index_x,
            queries,
            distances,
            query_radii,
            bandwidths,
            delta,
            build_time_s,
        )
        rows.append(row)
        print(
            f"[{dataset} {idx:02d}/{len(K_VALUES)}] K={k_value}: "
            f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
            f"F1={row['point_f1']:.3f} KDE={row['kde_abs_relative_error_vs_exact_r25']:.3f} "
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
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "mean_bandwidth": float(np.mean(bandwidths)),
        "delta": delta,
        "L": L_FIXED,
        "K_values": K_VALUES,
        "train_time_s": train_time_s,
        "encode_time_s": encode_time_s,
    }
    return pd.DataFrame(rows), meta


def plot_curves(df: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
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
    metrics = [
        ("point_precision", "Precision"),
        ("point_recall", "Recall"),
        ("point_f1", "F1"),
        ("kde_abs_relative_error_vs_exact_r25", "KDE error"),
        ("candidate_size", "Candidate size"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4.2))
    for dataset, part in df.groupby("dataset"):
        part = part.sort_values("K")
        for ax, (metric, title) in zip(axes, metrics):
            ax.plot(
                part["K"],
                part[metric],
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                color=colors.get(dataset),
                label=labels.get(dataset, dataset),
            )
    for ax, (_, title) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.28)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "K10_200_L20_sensitivity_curves.png", dpi=420)
    fig.savefig(figure_dir / "K10_200_L20_sensitivity_curves.pdf")
    plt.close(fig)


def summarize_common(df: pd.DataFrame) -> pd.DataFrame:
    common = (
        df.groupby("K")
        .agg(
            mean_precision=("point_precision", "mean"),
            mean_recall=("point_recall", "mean"),
            min_recall=("point_recall", "min"),
            mean_f1=("point_f1", "mean"),
            mean_kde_error=("kde_abs_relative_error_vs_exact_r25", "mean"),
            max_kde_error=("kde_abs_relative_error_vs_exact_r25", "max"),
            mean_candidate_size=("candidate_size", "mean"),
            mean_query_time_ms=("query_time_ms", "mean"),
        )
        .reset_index()
    )
    common["recall_feasible_all"] = common["min_recall"] >= 0.90
    return common


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
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
            elif col in {"K", "L"}:
                cells.append(str(int(value)))
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
        result_path = dataset_dir / "K10_200_L20_sensitivity.csv"
        if result_path.exists():
            print(f"Using existing K sensitivity results for {dataset}", flush=True)
            df = pd.read_csv(result_path)
            meta = {"dataset": dataset, "loaded_existing": True}
        else:
            print(f"=== Dataset: {dataset} ===", flush=True)
            df, meta = evaluate_dataset(dataset)
            df.to_csv(result_path, index=False)
        all_rows.append(df)
        metadata.append(meta)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(OUT_DIR / "K10_200_L20_sensitivity_all.csv", index=False)
    common = summarize_common(results)
    common.to_csv(OUT_DIR / "common_K10_200_L20_summary.csv", index=False)
    best_by_dataset = (
        results.sort_values(
            ["dataset", "point_f1", "point_recall", "point_precision"],
            ascending=[True, False, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
    )
    best_by_dataset.to_csv(OUT_DIR / "best_K_by_dataset.csv", index=False)
    plot_curves(results)

    top_common = common.sort_values(
        ["mean_f1", "mean_recall", "mean_precision"],
        ascending=[False, False, False],
    ).head(8)
    recall_common = common[common["recall_feasible_all"]].sort_values(
        ["mean_f1", "mean_recall", "mean_precision"],
        ascending=[False, False, False],
    ).head(8)

    config = {
        "datasets": DATASETS,
        "L": L_FIXED,
        "K_values": K_VALUES,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "bandwidth_mode": "h(q)=0.14R25(q)",
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta_mode": "delta=0.25*median_q R25(q)",
        "min_collisions": MIN_COLLISIONS,
        "hamming_probe": HAMMING_PROBE,
        "implementation_note": "This high-K experiment uses boolean code matrices instead of uint64 packing, because K>64 cannot be represented by the original uint64 code packer.",
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = [
        "# K=10..200 Sensitivity at L=20",
        "",
        "固定 `query-adaptive R25` 和 `h(q)=0.14R25(q)`，设置 `L=20`，扫描 `K=10,20,...,200`。",
        "",
        "## Best K by Dataset",
        "",
        markdown_table(
            best_by_dataset,
            [
                "dataset",
                "L",
                "K",
                "point_precision",
                "point_recall",
                "point_f1",
                "kde_abs_relative_error_vs_exact_r25",
                "candidate_size",
                "query_time_ms",
            ],
        ),
        "",
        "## Top Common Settings by Mean F1",
        "",
        markdown_table(
            top_common,
            [
                "K",
                "mean_precision",
                "mean_recall",
                "min_recall",
                "mean_f1",
                "mean_kde_error",
                "mean_candidate_size",
                "mean_query_time_ms",
            ],
        ),
        "",
        "## Top Common Settings with Minimum Recall >= 0.90",
        "",
        markdown_table(
            recall_common,
            [
                "K",
                "mean_precision",
                "mean_recall",
                "min_recall",
                "mean_f1",
                "mean_kde_error",
                "mean_candidate_size",
                "mean_query_time_ms",
            ],
        ),
        "",
        "## Outputs",
        "",
        "- `K10_200_L20_sensitivity_all.csv`",
        "- `common_K10_200_L20_summary.csv`",
        "- `best_K_by_dataset.csv`",
        "- `figures/K10_200_L20_sensitivity_curves.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved K=10..200, L=20 sensitivity to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
