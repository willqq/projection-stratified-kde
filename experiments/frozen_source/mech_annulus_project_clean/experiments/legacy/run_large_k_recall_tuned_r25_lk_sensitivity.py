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
from run_r25_k10_200_l20_paper_angle_lookup import (
    BANDWIDTH_FACTOR,
    DELTA_FACTOR,
    PrefixDistanceLookup,
    build_paper_sphere_table,
    encode_tables,
    hamming_threshold_from_angle,
    query_adaptive_r25,
)


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("r25_lk_large_k_recall_tuned_sensitivity")

L_VALUES = list(range(2, 21))
K_VALUES = [20, 30, 50, 100]
MAX_L = max(L_VALUES)
MAX_K = max(K_VALUES)
ANGLE_RECALL_ALPHA = 0.999
LAYER_PADDING = 1
MIN_HAMMING_FRACTIONS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
RADIUS_PERCENTILE = 25.0


def pack_prefix_key(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()


def build_prefix_lookups(index_bits: list[np.ndarray], bits: int) -> list[PrefixDistanceLookup]:
    return [PrefixDistanceLookup(table_bits, bits) for table_bits in index_bits]


def safe_layer_bounds(q_norm: float, radius: float, delta: float) -> tuple[int, int]:
    low = int(math.floor(max(0.0, q_norm - radius) / delta - 0.5 - LAYER_PADDING))
    high = int(math.ceil((q_norm + radius) / delta + 0.5 + LAYER_PADDING))
    low = max(low, 0)
    if high < low:
        high = low
    return low, high


def angle_from_norm(q_norm: float, p_norm: float, radius: float) -> float:
    if q_norm <= exp.EPS or p_norm <= exp.EPS:
        return math.pi
    cos_value = (q_norm**2 + p_norm**2 - radius**2) / max(
        2.0 * q_norm * p_norm,
        exp.EPS,
    )
    return math.acos(float(np.clip(cos_value, -1.0, 1.0)))


def layer_max_angle(q_norm: float, layer: int, delta: float, radius: float) -> float:
    if q_norm <= exp.EPS:
        return math.pi
    layer_low = max((float(layer) - 0.5) * delta, exp.EPS)
    layer_high = max((float(layer) + 0.5) * delta, exp.EPS)
    feasible_low = max(layer_low, q_norm - radius, exp.EPS)
    feasible_high = min(layer_high, q_norm + radius)
    if feasible_high < feasible_low:
        center = max(float(layer) * delta, exp.EPS)
        return angle_from_norm(q_norm, center, radius)

    candidates = [feasible_low, feasible_high]
    if q_norm > radius:
        critical = math.sqrt(max(q_norm**2 - radius**2, exp.EPS))
        if feasible_low <= critical <= feasible_high:
            candidates.append(critical)
    return max(angle_from_norm(q_norm, p_norm, radius) for p_norm in candidates)


def tuned_threshold(theta: float, bits: int, table_count: int, min_fraction: float) -> int:
    base = hamming_threshold_from_angle(theta, bits, table_count, ANGLE_RECALL_ALPHA)
    floor = int(math.ceil(float(min_fraction) * float(bits)))
    return min(max(base, floor), bits)


def tuned_ball_query(
    lookups: list[PrefixDistanceLookup],
    query_bits: list[np.ndarray],
    q_id: int,
    q_norm: float,
    radius: float,
    sphere_table: dict[int, np.ndarray],
    layer_masks: dict[int, np.ndarray],
    delta: float,
    n_index: int,
    table_count: int,
    code_bits: int,
    min_hamming_fraction: float,
) -> tuple[np.ndarray, float, float]:
    low, high = safe_layer_bounds(q_norm, radius, delta)
    selected = np.zeros(n_index, dtype=bool)
    thresholds = []
    theta_values = []

    for layer in range(low, high + 1):
        if layer not in sphere_table:
            continue
        theta = layer_max_angle(q_norm, layer, delta, radius)
        threshold = tuned_threshold(theta, code_bits, table_count, min_hamming_fraction)
        thresholds.append(threshold)
        theta_values.append(theta)

        layer_hash_selected = np.zeros(n_index, dtype=bool)
        for table_id in range(table_count):
            ids = lookups[table_id].ids_within_hamming(
                query_bits[table_id][q_id],
                threshold,
            )
            if len(ids):
                layer_hash_selected[ids] = True
        selected |= layer_hash_selected & layer_masks[layer]

    return (
        np.flatnonzero(selected).astype(np.int32),
        float(np.mean(thresholds)) if thresholds else 0.0,
        float(np.mean(theta_values)) if theta_values else 0.0,
    )


def evaluate_setting(
    dataset: str,
    table_count: int,
    code_bits: int,
    min_hamming_fraction: float,
    lookups: list[PrefixDistanceLookup],
    query_bits: list[np.ndarray],
    index_x: np.ndarray,
    queries: np.ndarray,
    distances: np.ndarray,
    query_radii: np.ndarray,
    bandwidths: np.ndarray,
    delta: float,
    sphere_table: dict[int, np.ndarray],
    layer_masks: dict[int, np.ndarray],
    query_norms: np.ndarray,
    build_time_s: float,
) -> dict[str, float | int | str]:
    precision_values = []
    recall_values = []
    f1_values = []
    candidate_sizes = []
    exact_r25_sizes = []
    kde_errors = []
    weighted_recalls = []
    thresholds = []
    theta_values = []
    query_times = []

    for q_id, _ in enumerate(queries):
        radius = float(query_radii[q_id])
        bandwidth = float(bandwidths[q_id])
        d = distances[q_id]
        true_ids = np.flatnonzero(d <= radius).astype(np.int32)
        if len(true_ids) == 0:
            continue

        weights = exp.gaussian_kernel(d, bandwidth)
        true_kde = float(weights[true_ids].sum())

        start = time.perf_counter()
        approx_ids, mean_threshold, mean_theta = tuned_ball_query(
            lookups,
            query_bits,
            q_id,
            float(query_norms[q_id]),
            radius,
            sphere_table,
            layer_masks,
            delta,
            len(index_x),
            table_count,
            code_bits,
            min_hamming_fraction,
        )
        query_times.append(time.perf_counter() - start)
        thresholds.append(mean_threshold)
        theta_values.append(mean_theta)

        approx_kde = float(weights[approx_ids].sum()) if len(approx_ids) else 0.0
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
        tp_weight = float(weights[tp].sum()) if len(tp) else 0.0
        weighted_recalls.append(tp_weight / max(true_kde, exp.EPS))

    return {
        "dataset": dataset,
        "L": int(table_count),
        "K": int(code_bits),
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta": float(delta),
        "angle_alpha": ANGLE_RECALL_ALPHA,
        "layer_padding": LAYER_PADDING,
        "min_hamming_fraction": float(min_hamming_fraction),
        "min_hamming_floor": int(math.ceil(min_hamming_fraction * code_bits)),
        "point_precision": float(np.mean(precision_values)),
        "point_recall": float(np.mean(recall_values)),
        "point_f1": float(np.mean(f1_values)),
        "kernel_weighted_recall": float(np.mean(weighted_recalls)),
        "kde_abs_relative_error_vs_exact_r25": float(np.mean(kde_errors)),
        "candidate_size": float(np.mean(candidate_sizes)),
        "exact_r25_candidate_size": float(np.mean(exact_r25_sizes)),
        "candidate_expansion_ratio": float(
            np.mean(np.asarray(candidate_sizes) / np.maximum(exact_r25_sizes, 1))
        ),
        "mean_hamming_threshold": float(np.mean(thresholds)),
        "mean_theta_deg": float(math.degrees(np.mean(theta_values))),
        "query_time_ms": float(np.mean(query_times) * 1000.0),
        "build_time_s": float(build_time_s),
    }


def expected_rows() -> int:
    return len(L_VALUES) * len(K_VALUES) * len(MIN_HAMMING_FRACTIONS)


def load_complete_result(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == expected_rows():
        return df
    return None


def save_dataset_rows(path: Path, rows: pd.DataFrame) -> None:
    rows = rows.sort_values(["dataset", "K", "min_hamming_fraction", "L"])
    rows = rows.drop_duplicates(["dataset", "L", "K", "min_hamming_fraction"], keep="last")
    rows.to_csv(path, index=False)


def evaluate_dataset(dataset: str, result_path: Path) -> tuple[pd.DataFrame, dict]:
    existing = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    completed = set()
    if not existing.empty:
        completed = {
            (int(row.L), int(row.K), float(row.min_hamming_fraction))
            for row in existing.itertuples()
        }

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
        MAX_L,
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

    norms = np.linalg.norm(index_x, axis=1)
    sphere_table, _ = build_paper_sphere_table(norms, delta)
    layer_masks = {}
    for layer, ids in sphere_table.items():
        mask = np.zeros(len(index_x), dtype=bool)
        mask[ids] = True
        layer_masks[layer] = mask
    query_norms = np.linalg.norm(queries, axis=1)

    rows = existing.to_dict("records") if not existing.empty else []
    total = expected_rows()
    done = len(completed)
    for code_bits in K_VALUES:
        lookups = build_prefix_lookups(index_bits, int(code_bits))
        for min_fraction in MIN_HAMMING_FRACTIONS:
            for table_count in L_VALUES:
                key = (int(table_count), int(code_bits), float(min_fraction))
                if key in completed:
                    continue
                row = evaluate_setting(
                    dataset,
                    int(table_count),
                    int(code_bits),
                    float(min_fraction),
                    lookups,
                    query_bits,
                    index_x,
                    queries,
                    distances,
                    query_radii,
                    bandwidths,
                    delta,
                    sphere_table,
                    layer_masks,
                    query_norms,
                    build_time_s,
                )
                rows.append(row)
                completed.add(key)
                done += 1
                save_dataset_rows(result_path, pd.DataFrame(rows))
                print(
                    f"[{dataset} {done:04d}/{total}] "
                    f"L={table_count} K={code_bits} minH={min_fraction:.2f}: "
                    f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                    f"F1={row['point_f1']:.3f} "
                    f"KDE={row['kde_abs_relative_error_vs_exact_r25']:.3f} "
                    f"C={row['candidate_size']:.1f} I={row['mean_hamming_threshold']:.2f} "
                    f"T={row['query_time_ms']:.3f}ms",
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
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "angle_alpha": ANGLE_RECALL_ALPHA,
        "layer_padding": LAYER_PADDING,
        "min_hamming_fractions": MIN_HAMMING_FRACTIONS,
        "train_time_s": train_time_s,
        "encode_time_s": encode_time_s,
    }
    return pd.DataFrame(rows), meta


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kde_quality"] = (
        1.0 - df["kde_abs_relative_error_vs_exact_r25"]
    ).clip(lower=0.0, upper=1.0)
    df["recall_first_score"] = (
        0.55 * df["point_recall"]
        + 0.25 * df["kde_quality"]
        + 0.10 * df["point_f1"]
        - 0.05 * np.log1p(df["candidate_expansion_ratio"])
        - 0.05 * np.log1p(df["query_time_ms"])
    )
    return df


def summarize_outputs(results: pd.DataFrame, metadata: list[dict]) -> None:
    results = add_scores(results)
    results.to_csv(OUT_DIR / "large_k_recall_tuned_all.csv", index=False)

    best_by_dataset_k = (
        results.sort_values(
            [
                "dataset",
                "K",
                "point_recall",
                "kde_abs_relative_error_vs_exact_r25",
                "candidate_size",
            ],
            ascending=[True, True, False, True, True],
        )
        .groupby(["dataset", "K"], as_index=False)
        .head(1)
    )
    best_by_dataset_k.to_csv(OUT_DIR / "best_recall_by_dataset_k.csv", index=False)

    best_balanced_by_dataset_k = (
        results.sort_values(
            [
                "dataset",
                "K",
                "recall_first_score",
                "point_recall",
                "kde_abs_relative_error_vs_exact_r25",
            ],
            ascending=[True, True, False, False, True],
        )
        .groupby(["dataset", "K"], as_index=False)
        .head(1)
    )
    best_balanced_by_dataset_k.to_csv(
        OUT_DIR / "best_recall_first_score_by_dataset_k.csv",
        index=False,
    )

    common = (
        results.groupby(["L", "K", "min_hamming_fraction"], as_index=False)
        .agg(
            mean_precision=("point_precision", "mean"),
            mean_recall=("point_recall", "mean"),
            min_recall=("point_recall", "min"),
            mean_f1=("point_f1", "mean"),
            mean_kde_error=("kde_abs_relative_error_vs_exact_r25", "mean"),
            max_kde_error=("kde_abs_relative_error_vs_exact_r25", "max"),
            mean_candidate_size=("candidate_size", "mean"),
            mean_query_time_ms=("query_time_ms", "mean"),
            mean_recall_first_score=("recall_first_score", "mean"),
        )
    )
    common.to_csv(OUT_DIR / "common_large_k_recall_tuned_summary.csv", index=False)

    best_common_by_k = (
        common.sort_values(
            [
                "K",
                "mean_recall",
                "mean_kde_error",
                "mean_candidate_size",
                "mean_query_time_ms",
            ],
            ascending=[True, False, True, True, True],
        )
        .groupby("K", as_index=False)
        .head(1)
    )
    best_common_by_k.to_csv(OUT_DIR / "best_common_recall_by_k.csv", index=False)

    plot_dir = OUT_DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset, part in best_by_dataset_k.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(part["K"], part["point_recall"], marker="o", label="best recall")
        ax.plot(
            part["K"],
            part["kde_abs_relative_error_vs_exact_r25"],
            marker="s",
            label="KDE error",
        )
        ax.set_title(f"{dataset}: tuned large-K recall")
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_best_recall_by_k.png", dpi=300)
        plt.close(fig)

    def format_value(value: object, column: str) -> str:
        if column == "dataset":
            return str(value)
        if column in {"L", "K", "min_hamming_floor"}:
            return str(int(value))
        return f"{float(value):.4f}"

    def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in frame.iterrows():
            lines.append(
                "| " + " | ".join(format_value(row[col], col) for col in columns) + " |"
            )
        return "\n".join(lines)

    dataset_cols = [
        "dataset",
        "K",
        "L",
        "min_hamming_fraction",
        "min_hamming_floor",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "query_time_ms",
    ]
    common_cols = [
        "K",
        "L",
        "min_hamming_fraction",
        "mean_precision",
        "mean_recall",
        "min_recall",
        "mean_f1",
        "mean_kde_error",
        "mean_candidate_size",
        "mean_query_time_ms",
    ]

    summary = [
        "# 大 K 召回调参试验",
        "",
        "本实验仍使用 `query-adaptive R25` 和 `h(q)=0.14R25(q)`，没有改成更大的半径或带宽。",
        "",
        "## 调参方式",
        "",
        f"- `angle_alpha={ANGLE_RECALL_ALPHA}`，比原来的 `0.95` 更偏召回。",
        f"- 球面层边界增加 `layer_padding={LAYER_PADDING}`，减少径向层漏召回。",
        "- Hamming 阈值使用 `max(角度公式阈值, ceil(min_hamming_fraction*K))`。",
        f"- 扫描 `min_hamming_fraction={MIN_HAMMING_FRACTIONS}`。",
        f"- 大 K 档：`K={K_VALUES}`；哈希表数量仍扫 `L=2..20`。",
        "",
        "## 各数据集每个 K 的最高召回",
        "",
        markdown_table(best_by_dataset_k, dataset_cols),
        "",
        "## 每个 K 的公共最高平均召回组合",
        "",
        markdown_table(best_common_by_k, common_cols),
        "",
        "## 输出文件",
        "",
        "- `large_k_recall_tuned_all.csv`",
        "- `best_recall_by_dataset_k.csv`",
        "- `best_recall_first_score_by_dataset_k.csv`",
        "- `common_large_k_recall_tuned_summary.csv`",
        "- `best_common_recall_by_k.csv`",
    ]
    (OUT_DIR / "summary_zh.md").write_text("\n".join(summary), encoding="utf-8")

    config = {
        "datasets": DATASETS,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "bandwidth_mode": "h(q)=0.14R25(q)",
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta_mode": "delta=0.25*median_q R25(q)",
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "angle_alpha": ANGLE_RECALL_ALPHA,
        "layer_padding": LAYER_PADDING,
        "min_hamming_fractions": MIN_HAMMING_FRACTIONS,
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metadata = []
    for dataset in DATASETS:
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result_path = dataset_dir / "large_k_recall_tuned.csv"
        complete = load_complete_result(result_path)
        if complete is not None:
            print(f"Using existing tuned large-K results for {dataset}", flush=True)
            df = complete
            meta = {"dataset": dataset, "loaded_existing": True}
        else:
            print(f"=== Dataset: {dataset} ===", flush=True)
            df, meta = evaluate_dataset(dataset, result_path)
        all_rows.append(df)
        metadata.append(meta)

    summarize_outputs(pd.concat(all_rows, ignore_index=True), metadata)
    print(f"Saved tuned large-K recall results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
