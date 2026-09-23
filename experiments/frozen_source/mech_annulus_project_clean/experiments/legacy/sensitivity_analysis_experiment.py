from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler


SEED = 20260528
BASE_PARAMS = {
    "L": 8,
    "K": 8,
    "delta": 1.0,
    "radius_factor": 1.0,
    "bandwidth_factor": 1.0,
}

SWEEPS = {
    "hash_tables_L": ("L", [1, 2, 4, 8, 16]),
    "hash_code_length_K": ("K", [4, 6, 8, 10, 12, 14]),
    "sphere_interval_delta": ("delta", [0.5, 0.75, 1.0, 1.5, 2.0]),
    "annulus_radius": ("radius_factor", [0.5, 0.75, 1.0, 1.25, 1.5]),
    "kernel_bandwidth": ("bandwidth_factor", [0.5, 0.75, 1.0, 1.5, 2.0]),
}


@dataclass(frozen=True)
class Setting:
    group: str
    parameter: str
    value: float
    L: int
    K: int
    delta: float
    radius_factor: float
    bandwidth_factor: float


def min_max_scale(x: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return np.nan_to_num(scaler.fit_transform(x)).astype(np.float32)


def load_dataset(name: str) -> np.ndarray:
    name = name.lower()
    if name == "isolet":
        df = pd.read_csv("data/isolet/isolet1+2+3+4.data", header=None)
        x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    elif name == "cifar10":
        rows = []
        with open("data/cifar-10-batches-py/data_batch_1", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        rows.append(batch[b"data"].astype(np.float32))
        x = np.vstack(rows)
    elif name == "cifar10_gist512":
        x = scio.loadmat("data/cifar10-Gist512/Cifar10-Gist512.mat")["X"].astype(
            np.float32
        )
    elif name == "amazon":
        data, _ = arff.loadarff(
            "data/Amazon_initial_50_30_10000/"
            "Amazon_initial_50_30_10000.arff"
        )
        df = pd.DataFrame(data).drop(columns=["class_duplicate"])
        x = df.to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return min_max_scale(x)


def sample_index_and_queries(
    x: np.ndarray, n_index: int, n_queries: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    needed = min(len(x), n_index + n_queries)
    sample = rng.choice(len(x), size=needed, replace=False)
    x_sample = x[sample]
    n_index = min(n_index, needed - n_queries)
    if n_index <= 0:
        raise ValueError("Not enough data for the requested index/query split.")
    return x_sample[:n_index], x_sample[n_index : n_index + n_queries]


def compute_distance_matrix(index_x: np.ndarray, queries: np.ndarray) -> np.ndarray:
    distances = []
    for q in queries:
        distances.append(np.linalg.norm(index_x - q, axis=1))
    return np.vstack(distances).astype(np.float32)


class AngularLSHIndex:
    def __init__(
        self,
        x: np.ndarray,
        L: int,
        K: int,
        rng: np.random.Generator,
        hamming_probe: int = 1,
    ):
        self.x = x
        self.L = L
        self.K = K
        self.hamming_probe = hamming_probe
        self.powers = (1 << np.arange(K, dtype=np.uint64)).astype(np.uint64)
        self.projections = rng.normal(size=(L, K, x.shape[1])).astype(np.float32)
        norms = np.linalg.norm(self.projections, axis=2, keepdims=True)
        self.projections = self.projections / np.maximum(norms, 1e-12)
        self.tables = []
        self.build_time_s = 0.0
        self._build()

    def _build(self) -> None:
        start = time.perf_counter()
        for table_id in range(self.L):
            projections = self.projections[table_id]
            bits = (self.x @ projections.T) >= 0
            codes = bits.astype(np.uint64) @ self.powers
            table = defaultdict(list)
            for idx, code in enumerate(codes):
                table[int(code)].append(idx)
            self.tables.append(table)
        self.build_time_s = time.perf_counter() - start

    def _hash_query(self, q: np.ndarray, table_id: int) -> int:
        bits = (self.projections[table_id] @ q) >= 0
        return int(bits.astype(np.uint64) @ self.powers)

    def _neighbor_codes(self, code: int):
        yield code
        if self.hamming_probe >= 1:
            for bit in range(self.K):
                yield code ^ (1 << bit)
        if self.hamming_probe >= 2:
            for bit_a in range(self.K):
                for bit_b in range(bit_a + 1, self.K):
                    yield code ^ (1 << bit_a) ^ (1 << bit_b)

    def query(self, q: np.ndarray) -> set[int]:
        candidates = set()
        for table_id, table in enumerate(self.tables):
            code = self._hash_query(q, table_id)
            for neighbor_code in self._neighbor_codes(code):
                candidates.update(table.get(neighbor_code, []))
        return candidates


def build_sphere_table(norms: np.ndarray, delta: float) -> dict[int, np.ndarray]:
    buckets = defaultdict(list)
    layers = np.floor(norms / delta).astype(int)
    for idx, layer in enumerate(layers):
        buckets[int(layer)].append(idx)
    return {layer: np.asarray(indices, dtype=np.int32) for layer, indices in buckets.items()}


def sphere_candidates(
    sphere_table: dict[int, np.ndarray], q_norm: float, radius: float, delta: float
) -> set[int]:
    low = int(math.floor(max(0.0, q_norm - radius) / delta))
    high = int(math.ceil((q_norm + radius) / delta))
    candidates = []
    for layer in range(low, high + 1):
        if layer in sphere_table:
            candidates.append(sphere_table[layer])
    if not candidates:
        return set()
    return set(np.concatenate(candidates).tolist())


def gaussian_values(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-(distances**2) / (2.0 * bandwidth**2))


def evaluate_setting(
    setting: Setting,
    index_x: np.ndarray,
    queries: np.ndarray,
    distance_matrix: np.ndarray,
    base_radii: np.ndarray,
    base_bandwidth: float,
    lsh_index: AngularLSHIndex,
) -> dict:
    norms = np.linalg.norm(index_x, axis=1)
    query_norms = np.linalg.norm(queries, axis=1)
    sphere_start = time.perf_counter()
    sphere_table = build_sphere_table(norms, setting.delta)
    sphere_build_time_s = time.perf_counter() - sphere_start

    precision_values = []
    recall_values = []
    signed_bias_values = []
    abs_bias_values = []
    candidate_sizes = []
    query_times = []

    bandwidth = base_bandwidth * setting.bandwidth_factor

    for q_id, q in enumerate(queries):
        radius = base_radii[q_id] * setting.radius_factor
        distances = distance_matrix[q_id]
        true_set = set(np.flatnonzero(distances <= radius).tolist())

        start = time.perf_counter()
        hash_set = lsh_index.query(q)
        sphere_set = sphere_candidates(
            sphere_table, float(query_norms[q_id]), float(radius), setting.delta
        )
        retrieved = hash_set & sphere_set
        if retrieved and query_norms[q_id] > 0:
            prefilter_indices = np.fromiter(retrieved, dtype=np.int32)
            denom = query_norms[q_id] * norms[prefilter_indices]
            valid = denom > 1e-12
            cosines = np.zeros(len(prefilter_indices), dtype=np.float32)
            cosines[valid] = (
                index_x[prefilter_indices[valid]] @ q / denom[valid]
            )
            cosines = np.clip(cosines, -1.0, 1.0)
            angles = np.arccos(cosines)
            angle_limit = math.asin(min(0.99, radius / max(query_norms[q_id], 1e-12)))
            retrieved = set(prefilter_indices[angles <= angle_limit].tolist())
        if retrieved:
            retrieved_indices = np.fromiter(retrieved, dtype=np.int32)
        else:
            retrieved_indices = np.asarray([], dtype=np.int32)

        exact_density = float(np.mean(gaussian_values(distances, bandwidth)))
        if len(retrieved_indices) > 0:
            quantized_distances = (
                np.round(distances[retrieved_indices] / setting.delta) * setting.delta
            )
            approx_density = float(
                np.mean(gaussian_values(quantized_distances, bandwidth))
            )
        else:
            approx_density = 0.0
        query_times.append(time.perf_counter() - start)

        tp = len(true_set & retrieved)
        precision_values.append(tp / len(retrieved) if retrieved else 0.0)
        recall_values.append(tp / len(true_set) if true_set else 0.0)
        rel_bias = (approx_density - exact_density) / max(abs(exact_density), 1e-12)
        signed_bias_values.append(rel_bias)
        abs_bias_values.append(abs(rel_bias))
        candidate_sizes.append(len(retrieved))

    return {
        "group": setting.group,
        "parameter": setting.parameter,
        "value": setting.value,
        "L": setting.L,
        "K": setting.K,
        "delta": setting.delta,
        "radius_factor": setting.radius_factor,
        "bandwidth_factor": setting.bandwidth_factor,
        "bandwidth": bandwidth,
        "accuracy_precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "signed_relative_bias": float(np.mean(signed_bias_values)),
        "abs_relative_bias": float(np.mean(abs_bias_values)),
        "query_time_ms": float(np.mean(query_times) * 1000.0),
        "build_time_s": float(lsh_index.build_time_s + sphere_build_time_s),
        "candidate_size": float(np.mean(candidate_sizes)),
        "precision_std": float(np.std(precision_values)),
        "recall_std": float(np.std(recall_values)),
        "abs_bias_std": float(np.std(abs_bias_values)),
    }


def build_settings() -> list[Setting]:
    settings = []
    for group, (parameter, values) in SWEEPS.items():
        for value in values:
            params = dict(BASE_PARAMS)
            params[parameter] = value
            settings.append(
                Setting(
                    group=group,
                    parameter=parameter,
                    value=float(value),
                    L=int(params["L"]),
                    K=int(params["K"]),
                    delta=float(params["delta"]),
                    radius_factor=float(params["radius_factor"]),
                    bandwidth_factor=float(params["bandwidth_factor"]),
                )
            )
    return settings


def plot_group(summary: pd.DataFrame, group: str, out_dir: Path) -> None:
    sub = summary[summary["group"] == group].sort_values("value")
    x = sub["value"].to_numpy()
    x_label = sub["parameter"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160)
    axes = axes.ravel()

    series = [
        ("Accuracy / Precision", "accuracy_precision", "tab:blue"),
        ("Recall", "recall", "tab:green"),
        ("Abs. Relative Bias", "abs_relative_bias", "tab:red"),
    ]
    for ax, (title, column, color) in zip(axes[:3], series):
        ax.plot(x, sub[column].to_numpy(), marker="o", color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)
        if column != "abs_relative_bias":
            ax.set_ylim(0, 1.05)

    ax = axes[3]
    ax.plot(
        x,
        sub["query_time_ms"].to_numpy(),
        marker="o",
        color="tab:orange",
        linewidth=2,
        label="Query ms",
    )
    ax.set_title("Time")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Query time (ms)")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        sub["build_time_s"].to_numpy(),
        marker="s",
        color="tab:purple",
        linestyle="--",
        linewidth=1.8,
        label="Build s",
    )
    ax2.set_ylabel("Build time (s)")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best", fontsize=8)

    fig.suptitle(f"Sensitivity: {group}", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / f"{group}.png")
    plt.close(fig)


def plot_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("accuracy_precision", "Accuracy / Precision"),
        ("recall", "Recall"),
        ("abs_relative_bias", "Abs. Relative Bias"),
        ("query_time_ms", "Query Time (ms)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 12), dpi=160)
    for ax, (column, title) in zip(axes, metrics):
        labels = []
        values = []
        for group in SWEEPS:
            sub = summary[summary["group"] == group].sort_values("value")
            for _, row in sub.iterrows():
                labels.append(f"{row['parameter']}={row['value']:g}")
                values.append(row[column])
        ax.bar(np.arange(len(values)), values, color="tab:blue", alpha=0.75)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if column in {"accuracy_precision", "recall"}:
            ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_overview.png")
    plt.close(fig)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def make_report(summary: pd.DataFrame, config: dict, out_dir: Path) -> None:
    best_recall = summary.loc[summary["recall"].idxmax()]
    best_precision = summary.loc[summary["accuracy_precision"].idxmax()]
    best_bias = summary.loc[summary["abs_relative_bias"].idxmin()]
    fastest = summary.loc[summary["query_time_ms"].idxmin()]

    norm_time = summary["query_time_ms"] / max(summary["query_time_ms"].max(), 1e-12)
    norm_bias = summary["abs_relative_bias"] / max(summary["abs_relative_bias"].max(), 1e-12)
    score = (
        0.35 * summary["recall"]
        + 0.30 * summary["accuracy_precision"]
        - 0.25 * norm_bias
        - 0.10 * norm_time
    )
    balanced = summary.loc[score.idxmax()]

    lines = [
        "# 敏感性分析实验结论",
        "",
        "## 实验设置",
        "",
        f"- 数据集：{config['dataset']}",
        f"- 索引数据量：{config['n_index']}",
        f"- 查询点数量：{config['n_queries']}",
        f"- 距离半径基准：每个查询点到索引集距离的第 {config['radius_percentile']} 分位数",
        f"- 核带宽基准：上述半径基准的中位数，数值为 {config['base_bandwidth']:.6f}",
        "- 准确率定义：候选集中真实邻居所占比例，即 precision",
        "- 召回率定义：真实邻居中被候选集覆盖的比例",
        "- 偏差定义：候选集和 delta 距离量化得到的近似 KDE 与精确 KDE 的相对偏差，图中使用绝对相对偏差",
        "",
        "## 总体结果",
        "",
        (
            f"- 最高召回率：{pct(best_recall['recall'])}，出现在 "
            f"{best_recall['parameter']}={best_recall['value']:g}。"
        ),
        (
            f"- 最高准确率：{pct(best_precision['accuracy_precision'])}，出现在 "
            f"{best_precision['parameter']}={best_precision['value']:g}。"
        ),
        (
            f"- 最小绝对相对偏差：{pct(best_bias['abs_relative_bias'])}，出现在 "
            f"{best_bias['parameter']}={best_bias['value']:g}。"
        ),
        (
            f"- 最低平均查询时间：{fastest['query_time_ms']:.3f} ms，出现在 "
            f"{fastest['parameter']}={fastest['value']:g}。"
        ),
        (
            f"- 综合平衡设置建议：{balanced['parameter']}={balanced['value']:g} "
            f"(L={int(balanced['L'])}, K={int(balanced['K'])}, "
            f"delta={balanced['delta']:g}, radius_factor={balanced['radius_factor']:g}, "
            f"bandwidth_factor={balanced['bandwidth_factor']:g})。"
        ),
        "",
        "## 分参数结论",
        "",
    ]

    for group in SWEEPS:
        sub = summary[summary["group"] == group].sort_values("value")
        first = sub.iloc[0]
        last = sub.iloc[-1]
        local_best = sub.loc[
            (
                0.4 * sub["recall"]
                + 0.3 * sub["accuracy_precision"]
                - 0.2
                * sub["abs_relative_bias"]
                / max(sub["abs_relative_bias"].max(), 1e-12)
                - 0.1 * sub["query_time_ms"] / max(sub["query_time_ms"].max(), 1e-12)
            ).idxmax()
        ]
        lines.extend(
            [
                f"### {group}",
                "",
                (
                    f"- 从 {first['value']:g} 到 {last['value']:g}，准确率由 "
                    f"{pct(first['accuracy_precision'])} 变为 "
                    f"{pct(last['accuracy_precision'])}，召回率由 "
                    f"{pct(first['recall'])} 变为 {pct(last['recall'])}。"
                ),
                (
                    f"- 绝对相对偏差由 {pct(first['abs_relative_bias'])} 变为 "
                    f"{pct(last['abs_relative_bias'])}，平均查询时间由 "
                    f"{first['query_time_ms']:.3f} ms 变为 "
                    f"{last['query_time_ms']:.3f} ms。"
                ),
                (
                    f"- 该参数的一维平衡取值为 {local_best['parameter']}="
                    f"{local_best['value']:g}。"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 可用于论文的表述",
            "",
            (
                "敏感性分析表明，哈希表数量、哈希码长度和圆环半径是候选集规模、"
                "准确率与召回率的主要敏感因素。总体上，增大哈希表数量更有利于提高"
                "召回率，但会增加构建和查询代价；增大哈希码长度通常会缩小候选集，"
                "但过长时会损失召回率，实际查询时间还会受到哈希计算和实现开销影响。"
                "圆环半径扩大后候选集覆盖更充分，"
                "KDE 偏差下降，但候选集规模和查询时间上升。球面划分间隔 delta 在本数据"
                "尺度下对检索准确率和召回率影响较弱，主要体现在距离量化误差和时间波动上。"
                "核带宽不改变候选集，因此主要影响密度估计偏差，对检索准确率和召回率影响较弱。"
            ),
        ]
    )

    (out_dir / "experiment_conclusions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="isolet")
    parser.add_argument("--n-index", type=int, default=2500)
    parser.add_argument("--n-queries", type=int, default=40)
    parser.add_argument("--radius-percentile", type=float, default=20.0)
    parser.add_argument("--hamming-probe", type=int, default=0)
    parser.add_argument("--out-dir", default="sensitivity_results")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = load_dataset(args.dataset)
    index_x, queries = sample_index_and_queries(x, args.n_index, args.n_queries, rng)
    distance_matrix = compute_distance_matrix(index_x, queries)
    base_radii = np.percentile(distance_matrix, args.radius_percentile, axis=1)
    base_bandwidth = float(np.median(base_radii))

    settings = build_settings()
    index_cache: dict[tuple[int, int], AngularLSHIndex] = {}
    rows = []

    for setting in settings:
        cache_key = (setting.L, setting.K)
        if cache_key not in index_cache:
            local_rng = np.random.default_rng(SEED + setting.L * 1000 + setting.K)
            index_cache[cache_key] = AngularLSHIndex(
                index_x,
                setting.L,
                setting.K,
                local_rng,
                hamming_probe=args.hamming_probe,
            )
        rows.append(
            evaluate_setting(
                setting,
                index_x,
                queries,
                distance_matrix,
                base_radii,
                base_bandwidth,
                index_cache[cache_key],
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    config = {
        "dataset": args.dataset,
        "n_index": len(index_x),
        "n_queries": len(queries),
        "radius_percentile": args.radius_percentile,
        "hamming_probe": args.hamming_probe,
        "base_bandwidth": base_bandwidth,
        "base_params": BASE_PARAMS,
        "sweeps": SWEEPS,
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for group in SWEEPS:
        plot_group(summary, group, out_dir)
    plot_overview(summary, out_dir)
    make_report(summary, config, out_dir)

    print(f"Saved results to {out_dir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
