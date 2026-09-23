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
from scipy.stats import binom

import mech_annulus_experiments as exp
from run_multidataset_requested_sensitivity import split_sizes


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("r25_k10_200_l20_paper_angle_lookup")

L_FIXED = 20
K_VALUES = list(range(10, 201, 10))
MAX_K = max(K_VALUES)
RADIUS_PERCENTILE = 25.0
BANDWIDTH_FACTOR = 0.14
DELTA_FACTOR = 0.25
ANGLE_RECALL_ALPHA = 0.95
NORM_FILTER_RULE = "| ||x|| - ||q|| | <= r(q)"
LOOKUP_RULE = (
    "distance-table hash-bucket lookup, union over L tables, "
    "intersect sphere layers, then exact norm-table filter"
)


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


@dataclass(frozen=True)
class NormTable:
    index_norms: np.ndarray
    sorted_norms: np.ndarray
    sorted_ids: np.ndarray

    @classmethod
    def from_norms(cls, norms: np.ndarray) -> "NormTable":
        index_norms = norms.astype(np.float32, copy=True)
        order = np.argsort(index_norms, kind="mergesort").astype(np.int32)
        return cls(
            index_norms=index_norms,
            sorted_norms=index_norms[order],
            sorted_ids=order,
        )

    def ids_within_radius(self, q_norm: float, radius: float) -> np.ndarray:
        # Triangle inequality: ||x-q|| <= r implies | ||x|| - ||q|| | <= r.
        low = float(q_norm) - float(radius) - exp.EPS
        high = float(q_norm) + float(radius) + exp.EPS
        start = int(np.searchsorted(self.sorted_norms, low, side="left"))
        stop = int(np.searchsorted(self.sorted_norms, high, side="right"))
        return self.sorted_ids[start:stop]

    def mask_within_radius(self, q_norm: float, radius: float) -> np.ndarray:
        mask = np.zeros(len(self.index_norms), dtype=bool)
        ids = self.ids_within_radius(q_norm, radius)
        if len(ids):
            mask[ids] = True
        return mask

    def filter_ids(self, ids: np.ndarray, q_norm: float, radius: float) -> np.ndarray:
        if len(ids) == 0:
            return ids
        keep = np.abs(self.index_norms[ids] - float(q_norm)) <= float(radius) + exp.EPS
        return ids[keep].astype(np.int32, copy=False)


def paper_layer_bounds(q_norm: float, radius: float, delta: float) -> tuple[int, int]:
    low = int(round(max(0.0, q_norm - radius) / delta))
    high = int(round((q_norm + radius) / delta))
    if high < low:
        high = low
    return low, high


def layer_angle_from_radius(q_norm: float, layer: int, delta: float, radius: float) -> float:
    if q_norm <= exp.EPS:
        return math.pi
    p_norm = max(float(layer) * delta, exp.EPS)
    cos_value = (q_norm**2 + p_norm**2 - radius**2) / max(
        2.0 * q_norm * p_norm,
        exp.EPS,
    )
    return math.acos(float(np.clip(cos_value, -1.0, 1.0)))


def hamming_threshold_from_angle(
    theta: float,
    bits: int,
    tables: int,
    alpha: float,
) -> int:
    bit_flip_probability = float(np.clip(theta / math.pi, 0.0, 1.0))
    target_single_table_probability = 1.0 - (1.0 - alpha) ** (1.0 / tables)
    threshold = int(
        binom.ppf(target_single_table_probability, bits, bit_flip_probability)
    )
    return min(max(threshold, 0), bits)


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


def paper_ball_query(
    lookups: list[PrefixDistanceLookup],
    query_bits: list[np.ndarray],
    q_id: int,
    q_norm: float,
    radius: float,
    sphere_table: dict[int, np.ndarray],
    layer_masks: dict[int, np.ndarray],
    norm_table: NormTable,
    delta: float,
    n_index: int,
) -> tuple[np.ndarray, float, float]:
    low, high = paper_layer_bounds(q_norm, radius, delta)
    selected = np.zeros(n_index, dtype=bool)
    norm_mask = norm_table.mask_within_radius(q_norm, radius)
    thresholds = []
    theta_values = []

    for layer in range(low, high + 1):
        layer_ids = sphere_table.get(layer)
        if layer_ids is None or len(layer_ids) == 0:
            continue
        theta = layer_angle_from_radius(q_norm, layer, delta, radius)
        threshold = hamming_threshold_from_angle(
            theta,
            lookups[0].bits,
            L_FIXED,
            ANGLE_RECALL_ALPHA,
        )
        thresholds.append(threshold)
        theta_values.append(theta)

        layer_hash_selected = np.zeros(n_index, dtype=bool)
        for table_id, lookup in enumerate(lookups):
            ids = lookup.ids_within_hamming(query_bits[table_id][q_id], threshold)
            if len(ids):
                layer_hash_selected[ids] = True
        selected |= layer_hash_selected & layer_masks[layer] & norm_mask

    return (
        np.flatnonzero(selected).astype(np.int32),
        float(np.mean(thresholds)) if thresholds else 0.0,
        float(np.mean(theta_values)) if theta_values else 0.0,
    )


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
    norm_table = NormTable.from_norms(norms)
    sphere_table, layers = build_paper_sphere_table(norms, delta)
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
        approx_ids, mean_threshold, mean_theta = paper_ball_query(
            lookups,
            query_bits,
            q_id,
            float(query_norms[q_id]),
            radius,
            sphere_table,
            layer_masks,
            norm_table,
            delta,
            len(index_x),
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
        "sphere_layer_rule": "round(norm/delta)",
        "theta_rule": "layer-wise cosine theorem",
        "norm_filter_rule": NORM_FILTER_RULE,
        "hamming_threshold_rule": "Eq. I_threshold with Binomial(K, theta/pi)",
        "lookup_rule": LOOKUP_RULE,
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
        "norm_filter_rule": NORM_FILTER_RULE,
        "lookup_rule": LOOKUP_RULE,
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
    fig.savefig(figure_dir / "K10_200_L20_paper_angle_lookup_curves.png", dpi=600)
    fig.savefig(figure_dir / "K10_200_L20_paper_angle_lookup_curves.pdf")
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


def has_current_norm_filter(df: pd.DataFrame) -> bool:
    return "norm_filter_rule" in df.columns and (df["norm_filter_rule"] == NORM_FILTER_RULE).all()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metadata = []
    for dataset in DATASETS:
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result_path = dataset_dir / "K10_200_L20_paper_angle_lookup.csv"
        if result_path.exists():
            cached = pd.read_csv(result_path)
            if has_current_norm_filter(cached):
                print(f"Using existing paper angle-lookup results for {dataset}", flush=True)
                df = cached
                meta = {
                    "dataset": dataset,
                    "loaded_existing": True,
                    "norm_filter_rule": NORM_FILTER_RULE,
                }
            else:
                print(
                    f"Existing paper angle-lookup results for {dataset} do not include "
                    "the current norm-table filter; recomputing.",
                    flush=True,
                )
                df, meta = evaluate_dataset(dataset)
                df.to_csv(result_path, index=False)
        else:
            print(f"=== Dataset: {dataset} ===", flush=True)
            df, meta = evaluate_dataset(dataset)
            df.to_csv(result_path, index=False)
        all_rows.append(df)
        metadata.append(meta)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(OUT_DIR / "K10_200_L20_paper_angle_lookup_all.csv", index=False)
    common = summarize_common(results)
    common.to_csv(OUT_DIR / "common_K10_200_L20_paper_angle_lookup_summary.csv", index=False)
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
        "sphere_layer_rule": "s=round(||p||/delta), fs=round((||q||-r)/delta), ls=round((||q||+r)/delta)",
        "theta_rule": "theta=arccos((||q||^2+||p_layer||^2-r^2)/(2||q||||p_layer||))",
        "norm_filter_rule": NORM_FILTER_RULE,
        "hamming_threshold_rule": "I=min I such that BinomialCDF(I; K, theta/pi)>1-(1-alpha)^(1/L)",
        "angle_recall_alpha": ANGLE_RECALL_ALPHA,
        "lookup_rule": LOOKUP_RULE,
        "table_merge": "union over L tables as in C_q(theta)",
        "implementation_note": "This high-K implementation packs prefix bits with np.packbits to support K>64; distance lookup is performed over hash-bucket codes rather than pointwise Hamming scans.",
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = [
        "# K=10..200 Sensitivity at L=20 with Paper Angle-Lookup Rule",
        "",
        "固定 `query-adaptive R25` 和 `h(q)=0.14R25(q)`，设置 `L=20`，扫描 `K=10,20,...,200`。",
        "",
        "本实验按照 `最新论文v2.md` 的流程计算：先由欧氏半径和球面层计算角度阈值，再依据 Eq. `I_threshold` 将角度转为 Hamming 距离阈值，最后通过距离表查找满足阈值的哈希桶，并与 Sphere Table 的径向层过滤结果相交。",
        f"新增 Norm Table 精确过滤：{NORM_FILTER_RULE}。该条件来自三角不等式，用于在近似环查询前去掉模长差距超过查询半径的候选点。",
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
        "- `K10_200_L20_paper_angle_lookup_all.csv`",
        "- `common_K10_200_L20_paper_angle_lookup_summary.csv`",
        "- `best_K_by_dataset.csv`",
        "- `figures/K10_200_L20_paper_angle_lookup_curves.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved paper angle-lookup K sensitivity to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
