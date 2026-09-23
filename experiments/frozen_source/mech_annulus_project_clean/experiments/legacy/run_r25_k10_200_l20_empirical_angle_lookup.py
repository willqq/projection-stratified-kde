from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
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
OUT_DIR = Path("r25_k10_200_l20_empirical_angle_lookup")

L_FIXED = 20
K_VALUES = list(range(10, 201, 10))
MAX_K = max(K_VALUES)
K_INDICES = [k - 1 for k in K_VALUES]
RADIUS_PERCENTILE = 25.0
BANDWIDTH_FACTOR = 0.14
DELTA_FACTOR = 0.25
ANGLE_RECALL_ALPHA = 0.95
CALIBRATION_PAIRS = 60000
MIN_CALIBRATION_PAIRS = 300


def query_adaptive_r25(distances: np.ndarray) -> np.ndarray:
    return np.percentile(distances, RADIUS_PERCENTILE, axis=1).astype(np.float32)


def encode_tables(
    model: exp.MultiEncoderContrastiveHash,
    x: np.ndarray,
) -> list[np.ndarray]:
    with torch.no_grad():
        encoded = model.encode_continuous(torch.from_numpy(x.astype(np.float32)))
    return [h.numpy().astype(np.float32) for h in encoded]


def pack_prefix_key(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()


def build_paper_sphere_table(norms: np.ndarray, delta: float) -> tuple[dict[int, np.ndarray], np.ndarray]:
    layers = np.rint(norms / delta).astype(np.int32)
    buckets: dict[int, list[int]] = {}
    for idx, layer in enumerate(layers):
        buckets.setdefault(int(layer), []).append(idx)
    return (
        {layer: np.asarray(ids, dtype=np.int32) for layer, ids in buckets.items()},
        layers,
    )


def paper_layer_bounds(q_norm: float, radius: float, delta: float) -> tuple[int, int]:
    low = int(math.floor(max(0.0, q_norm - radius) / delta - 0.5))
    high = int(math.ceil((q_norm + radius) / delta + 0.5))
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


def layer_max_angle_from_radius(
    q_norm: float,
    layer: int,
    delta: float,
    radius: float,
) -> float:
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


@dataclass
class EmpiricalThresholdCalibrator:
    sorted_angles: np.ndarray
    sorted_min_distances: np.ndarray
    bits_to_row: dict[int, int]
    alpha: float
    min_pairs: int
    cache: dict[tuple[int, int], int] = field(default_factory=dict)

    def threshold(self, theta: float, bits: int) -> int:
        row = self.bits_to_row[int(bits)]
        upper = int(np.searchsorted(self.sorted_angles, theta, side="right"))
        upper = min(max(upper, self.min_pairs), len(self.sorted_angles))
        cache_key = (row, upper)
        if cache_key in self.cache:
            return self.cache[cache_key]
        values = self.sorted_min_distances[row, :upper]
        q_index = int(math.ceil(self.alpha * len(values))) - 1
        q_index = min(max(q_index, 0), len(values) - 1)
        threshold = int(np.partition(values, q_index)[q_index])
        threshold = min(max(threshold, 0), bits)
        self.cache[cache_key] = threshold
        return threshold


def build_empirical_calibrator(
    train_x: np.ndarray,
    train_bits: list[np.ndarray],
    rng: np.random.Generator,
) -> EmpiricalThresholdCalibrator:
    n = len(train_x)
    pair_count = min(CALIBRATION_PAIRS, max(n * 20, MIN_CALIBRATION_PAIRS))
    left = rng.integers(0, n, size=pair_count, endpoint=False)
    right = rng.integers(0, n - 1, size=pair_count, endpoint=False)
    right = np.where(right >= left, right + 1, right)

    norms = np.linalg.norm(train_x, axis=1)
    denom = np.maximum(norms[left] * norms[right], exp.EPS)
    cosines = np.clip(np.sum(train_x[left] * train_x[right], axis=1) / denom, -1.0, 1.0)
    angles = np.arccos(cosines).astype(np.float32)

    min_distances = np.full((len(K_VALUES), pair_count), MAX_K, dtype=np.int16)
    for table_bits in train_bits:
        mismatches = table_bits[left, :MAX_K] != table_bits[right, :MAX_K]
        cumulative = np.cumsum(mismatches, axis=1, dtype=np.int16)[:, K_INDICES].T
        min_distances = np.minimum(min_distances, cumulative)

    order = np.argsort(angles)
    return EmpiricalThresholdCalibrator(
        sorted_angles=angles[order],
        sorted_min_distances=min_distances[:, order],
        bits_to_row={bits: idx for idx, bits in enumerate(K_VALUES)},
        alpha=ANGLE_RECALL_ALPHA,
        min_pairs=min(MIN_CALIBRATION_PAIRS, pair_count),
    )


@dataclass
class PrefixDistanceLookup:
    index_bits: np.ndarray
    bits: int
    hash_table: dict[bytes, np.ndarray] = field(init=False)
    unique_bits: np.ndarray = field(init=False)
    bucket_ids: list[np.ndarray] = field(init=False)
    distance_table_cache: dict[bytes, list[np.ndarray]] = field(default_factory=dict)
    selected_ids_cache: dict[tuple[bytes, int], np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grouped: dict[bytes, list[int]] = {}
        first_bits: dict[bytes, np.ndarray] = {}
        prefix = self.index_bits[:, : self.bits]
        for idx, row in enumerate(prefix):
            key = pack_prefix_key(row)
            grouped.setdefault(key, []).append(idx)
            if key not in first_bits:
                first_bits[key] = row.copy()
        self.hash_table = {
            key: np.asarray(ids, dtype=np.int32) for key, ids in grouped.items()
        }
        keys = list(self.hash_table.keys())
        self.unique_bits = np.asarray([first_bits[key] for key in keys], dtype=bool)
        self.bucket_ids = [self.hash_table[key] for key in keys]

    def distance_table_row(self, query_bits: np.ndarray) -> list[np.ndarray]:
        key = pack_prefix_key(query_bits[: self.bits])
        if key not in self.distance_table_cache:
            distances = np.count_nonzero(
                self.unique_bits != query_bits[: self.bits],
                axis=1,
            ).astype(np.int16)
            row = []
            for distance in range(self.bits + 1):
                matched_bucket_ids = np.flatnonzero(distances == distance)
                if len(matched_bucket_ids) == 0:
                    row.append(np.asarray([], dtype=np.int32))
                else:
                    row.append(
                        np.unique(
                            np.concatenate(
                                [self.bucket_ids[idx] for idx in matched_bucket_ids]
                            )
                        ).astype(np.int32)
                    )
            self.distance_table_cache[key] = row
        return self.distance_table_cache[key]

    def ids_within_hamming(self, query_bits: np.ndarray, threshold: int) -> np.ndarray:
        key = pack_prefix_key(query_bits[: self.bits])
        threshold = int(min(max(threshold, 0), self.bits))
        cache_key = (key, threshold)
        if cache_key in self.selected_ids_cache:
            return self.selected_ids_cache[cache_key]

        distance_row = self.distance_table_row(query_bits)
        selected_rows = [ids for ids in distance_row[: threshold + 1] if len(ids)]
        if not selected_rows:
            ids = np.asarray([], dtype=np.int32)
        else:
            ids = np.unique(np.concatenate(selected_rows)).astype(np.int32)
        self.selected_ids_cache[cache_key] = ids
        return ids


def build_prefix_lookups(index_bits: list[np.ndarray], bits: int) -> list[PrefixDistanceLookup]:
    return [PrefixDistanceLookup(table_bits, bits) for table_bits in index_bits]


def empirical_ball_query(
    lookups: list[PrefixDistanceLookup],
    calibrator: EmpiricalThresholdCalibrator,
    query_bits: list[np.ndarray],
    q_id: int,
    q_norm: float,
    radius: float,
    sphere_table: dict[int, np.ndarray],
    layer_masks: dict[int, np.ndarray],
    delta: float,
    n_index: int,
    bits: int,
) -> tuple[np.ndarray, float, float]:
    low, high = paper_layer_bounds(q_norm, radius, delta)
    selected = np.zeros(n_index, dtype=bool)
    thresholds = []
    theta_values = []

    for layer in range(low, high + 1):
        layer_ids = sphere_table.get(layer)
        if layer_ids is None or len(layer_ids) == 0:
            continue
        theta = layer_max_angle_from_radius(q_norm, layer, delta, radius)
        threshold = calibrator.threshold(theta, bits)
        thresholds.append(threshold)
        theta_values.append(theta)

        layer_hash_selected = np.zeros(n_index, dtype=bool)
        for table_id, lookup in enumerate(lookups):
            ids = lookup.ids_within_hamming(query_bits[table_id][q_id], threshold)
            if len(ids):
                layer_hash_selected[ids] = True
        selected |= layer_hash_selected & layer_masks[layer]

    return (
        np.flatnonzero(selected).astype(np.int32),
        float(np.mean(thresholds)) if thresholds else 0.0,
        float(np.mean(theta_values)) if theta_values else 0.0,
    )


def evaluate_k(
    dataset: str,
    k_value: int,
    calibrator: EmpiricalThresholdCalibrator,
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
    sphere_table, _ = build_paper_sphere_table(norms, delta)
    layer_masks = {}
    for layer, ids in sphere_table.items():
        mask = np.zeros(len(index_x), dtype=bool)
        mask[ids] = True
        layer_masks[layer] = mask
    query_norms = np.linalg.norm(queries, axis=1)
    lookups = build_prefix_lookups(index_bits, k_value)

    precision_values = []
    recall_values = []
    f1_values = []
    candidate_sizes = []
    exact_r25_sizes = []
    kde_errors = []
    global_kde_errors = []
    weighted_recalls = []
    mean_thresholds = []
    mean_angles = []
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
        global_kde = float(weights.sum())

        start = time.perf_counter()
        approx_ids, mean_threshold, mean_theta = empirical_ball_query(
            lookups,
            calibrator,
            query_bits,
            q_id,
            float(query_norms[q_id]),
            radius,
            sphere_table,
            layer_masks,
            delta,
            len(index_x),
            k_value,
        )
        query_times.append(time.perf_counter() - start)
        mean_thresholds.append(mean_threshold)
        mean_angles.append(mean_theta)

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
        "angle_alpha": ANGLE_RECALL_ALPHA,
        "sphere_layer_rule": "round(norm/delta), safe floor/ceil query bounds",
        "theta_rule": "layer-wise maximum angle over layer interval",
        "hamming_threshold_rule": "empirical CDF of min Hamming distance over L tables",
        "lookup_rule": "distance-table hash-bucket lookup, union over L tables",
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
        "mean_hamming_threshold": float(np.mean(mean_thresholds)),
        "mean_theta_deg": float(math.degrees(np.mean(mean_angles))),
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
    train_encoded = encode_tables(model, train_x)
    index_encoded = encode_tables(model, index_x)
    query_encoded = encode_tables(model, queries)
    thresholds = [np.median(h, axis=0) for h in index_encoded]
    train_bits = [h >= thresholds[i] for i, h in enumerate(train_encoded)]
    index_bits = [h >= thresholds[i] for i, h in enumerate(index_encoded)]
    query_bits = [h >= thresholds[i] for i, h in enumerate(query_encoded)]
    calibrator = build_empirical_calibrator(train_x, train_bits, rng)
    encode_time_s = time.perf_counter() - encode_start
    build_time_s = train_time_s + encode_time_s

    rows = []
    for idx, k_value in enumerate(K_VALUES, start=1):
        row = evaluate_k(
            dataset,
            k_value,
            calibrator,
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
        "L": L_FIXED,
        "K_values": K_VALUES,
        "angle_alpha": ANGLE_RECALL_ALPHA,
        "calibration_pairs": CALIBRATION_PAIRS,
        "train_time_s": train_time_s,
        "encode_calibration_time_s": encode_time_s,
    }
    return pd.DataFrame(rows), meta


def plot_curves(df: pd.DataFrame) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
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
        ("mean_hamming_threshold", "Hamming threshold"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(25, 4.3))
    for dataset, part in df.groupby("dataset"):
        part = part.sort_values("K")
        for ax, (metric, title) in zip(axes, metrics):
            ax.plot(
                part["K"],
                part[metric],
                marker="o",
                linewidth=1.3,
                markersize=3.5,
                color=colors.get(dataset),
                label=labels.get(dataset, dataset),
            )
    for ax, (_, title) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "K10_200_L20_empirical_angle_lookup_curves.png", dpi=600)
    fig.savefig(figure_dir / "K10_200_L20_empirical_angle_lookup_curves.pdf")
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
            mean_hamming_threshold=("mean_hamming_threshold", "mean"),
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
            if isinstance(value, str):
                cells.append(value)
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
        result_path = dataset_dir / "K10_200_L20_empirical_angle_lookup.csv"
        if result_path.exists():
            print(f"Using existing empirical angle-lookup results for {dataset}", flush=True)
            df = pd.read_csv(result_path)
            meta = {"dataset": dataset, "loaded_existing": True}
        else:
            print(f"=== Dataset: {dataset} ===", flush=True)
            df, meta = evaluate_dataset(dataset)
            df.to_csv(result_path, index=False)
        all_rows.append(df)
        metadata.append(meta)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(OUT_DIR / "K10_200_L20_empirical_angle_lookup_all.csv", index=False)
    common = summarize_common(results)
    common.to_csv(OUT_DIR / "common_K10_200_L20_empirical_angle_lookup_summary.csv", index=False)
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
        "sphere_layer_rule": "s=round(||p||/delta), query bounds widened by half layer",
        "theta_rule": "maximum theta over each sphere-layer norm interval",
        "hamming_threshold_rule": "empirical alpha-quantile of min Hamming distance over L tables conditional on angle",
        "angle_recall_alpha": ANGLE_RECALL_ALPHA,
        "calibration_pairs": CALIBRATION_PAIRS,
        "lookup_rule": "retrieve bucket codes from distance tables within empirical Hamming threshold, merge hash-table buckets, and intersect sphere layers",
        "table_merge": "union over L tables",
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = [
        "# K=10..200 Sensitivity at L=20 with Empirical Angle-Lookup Rule",
        "",
        "固定 `query-adaptive R25` 和 `h(q)=0.14R25(q)`，设置 `L=20`，扫描 `K=10,20,...,200`。",
        "",
        "本实验修正了理论二项分布假设：对 MECH 学习得到的哈希码，使用训练样本对经验估计 `P(D_H<=I | theta)`，并按 `alpha=0.95` 选择经验 Hamming 阈值。查询阶段仍通过距离表查找满足阈值的哈希桶，并与 Sphere Table 径向层过滤结果相交。",
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
                "mean_hamming_threshold",
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
                "mean_hamming_threshold",
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
                "mean_hamming_threshold",
                "mean_query_time_ms",
            ],
        ),
        "",
        "## Outputs",
        "",
        "- `K10_200_L20_empirical_angle_lookup_all.csv`",
        "- `common_K10_200_L20_empirical_angle_lookup_summary.csv`",
        "- `best_K_by_dataset.csv`",
        "- `figures/K10_200_L20_empirical_angle_lookup_curves.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved empirical angle-lookup K sensitivity to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
