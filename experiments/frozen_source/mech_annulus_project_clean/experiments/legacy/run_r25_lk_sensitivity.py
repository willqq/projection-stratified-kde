from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech_annulus_experiments as exp
from run_multidataset_requested_sensitivity import split_sizes


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("r25_lk_sensitivity")

RADIUS_PERCENTILE = 25.0
BANDWIDTH_FACTOR = 0.14
DELTA_FACTOR = 0.25
L_VALUES = list(range(2, 13))
K_VALUES = list(range(2, 13))

BASE = {
    "L": 8,
    "K": 8,
    "hamming_probe": 2,
    "min_collisions": 2,
    "annulus_count": 1,
}


def query_adaptive_r25(distances: np.ndarray) -> np.ndarray:
    return np.percentile(distances, RADIUS_PERCENTILE, axis=1).astype(np.float32)


def evaluate_adaptive_r25(
    index: exp.HashIndexBase,
    index_x: np.ndarray,
    queries: np.ndarray,
    distances: np.ndarray,
    query_radii: np.ndarray,
    bandwidths: np.ndarray,
    delta: float,
    min_collisions: int,
    annulus_count: int,
) -> dict[str, float | str]:
    norms = np.linalg.norm(index_x, axis=1)
    layers = np.floor(norms / delta).astype(int)
    sphere_table = exp.build_sphere_table(norms, delta)
    query_norms = np.linalg.norm(queries, axis=1)
    all_ids = set(range(len(index_x)))
    query_codes = [tuple(index.query_codes(q)) for q in queries]

    point_precision = []
    point_recall = []
    point_f1 = []
    point_fp_ratio = []
    point_fn_ratio = []
    weighted_recall = []
    weighted_fp = []
    weighted_fn = []
    signed_bias = []
    kde_abs_error = []
    global_kde_abs_error = []
    exact_truncation_error = []
    candidate_expansion = []
    candidate_sizes = []
    exact_r25_sizes = []
    times = []
    filter_times = []
    kde_times = []
    diagnostics: dict[str, list[float]] = defaultdict(list)

    for q_id, q in enumerate(queries):
        d = distances[q_id]
        radius = float(query_radii[q_id])
        bandwidth = float(bandwidths[q_id])
        true = set(np.flatnonzero(d <= radius).tolist())
        if not true:
            continue

        weights = exp.gaussian_kernel(d, bandwidth)
        true_kde = float(weights[list(true)].sum())
        global_kde = float(weights.sum())
        exact_truncation_error.append(
            abs(true_kde - global_kde) / max(global_kde, exp.EPS)
        )

        first_candidates: set[int] = set()
        first_diag: dict[str, float] = {}
        repeated_times = []
        repeated_filter_times = []
        repeated_kde_times = []
        approx_kde = 0.0
        for _ in range(exp.TIMING_REPEATS):
            start = time.perf_counter()
            candidates, diag, _ = exp.approx_fixed_radius_partition_radius_first(
                index,
                list(query_codes[q_id]),
                q,
                float(query_norms[q_id]),
                radius,
                annulus_count,
                norms,
                layers,
                sphere_table,
                delta,
                "full",
                min_collisions,
                all_ids,
            )
            filter_done = time.perf_counter()
            if candidates:
                ids = np.fromiter(candidates, dtype=np.int32)
                current_kde = float(weights[ids].sum())
            else:
                current_kde = 0.0
            end = time.perf_counter()
            repeated_filter_times.append(filter_done - start)
            repeated_kde_times.append(end - filter_done)
            repeated_times.append(end - start)
            if not first_diag:
                first_candidates = candidates
                first_diag = diag
                approx_kde = current_kde

        times.append(float(np.median(repeated_times)))
        filter_times.append(float(np.median(repeated_filter_times)))
        kde_times.append(float(np.median(repeated_kde_times)))
        for key, value in first_diag.items():
            diagnostics[key].append(value)

        approx = first_candidates
        tp = true & approx
        fp = approx - true
        fn = true - approx

        precision = len(tp) / len(approx) if approx else 0.0
        recall = len(tp) / len(true)
        point_precision.append(precision)
        point_recall.append(recall)
        point_f1.append(2 * precision * recall / (precision + recall + exp.EPS))
        point_fp_ratio.append(len(fp) / len(true))
        point_fn_ratio.append(len(fn) / len(true))
        candidate_sizes.append(len(approx))
        exact_r25_sizes.append(len(true))
        candidate_expansion.append(len(approx) / max(len(true), 1))

        tp_w = float(weights[list(tp)].sum()) if tp else 0.0
        fp_w = float(weights[list(fp)].sum()) if fp else 0.0
        fn_w = float(weights[list(fn)].sum()) if fn else 0.0
        weighted_recall.append(tp_w / max(true_kde, exp.EPS))
        weighted_fp.append(fp_w / max(true_kde, exp.EPS))
        weighted_fn.append(fn_w / max(true_kde, exp.EPS))

        rel_bias = (approx_kde - true_kde) / max(true_kde, exp.EPS)
        signed_bias.append(rel_bias)
        kde_abs_error.append(abs(rel_bias))
        global_kde_abs_error.append(abs(approx_kde - global_kde) / max(global_kde, exp.EPS))

    return {
        "method": index.name,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta": float(delta),
        "min_collisions": int(min_collisions),
        "annulus_count": int(annulus_count),
        "point_precision": float(np.mean(point_precision)),
        "point_recall": float(np.mean(point_recall)),
        "point_f1": float(np.mean(point_f1)),
        "point_fp_ratio": float(np.mean(point_fp_ratio)),
        "point_fn_ratio": float(np.mean(point_fn_ratio)),
        "kernel_weighted_recall": float(np.mean(weighted_recall)),
        "kernel_weighted_fp_ratio": float(np.mean(weighted_fp)),
        "kernel_weighted_fn_ratio": float(np.mean(weighted_fn)),
        "signed_kde_bias_vs_exact_r25": float(np.mean(signed_bias)),
        "kde_abs_relative_error_vs_exact_r25": float(np.mean(kde_abs_error)),
        "kde_abs_relative_error_vs_global": float(np.mean(global_kde_abs_error)),
        "exact_r25_truncation_error_vs_global": float(np.mean(exact_truncation_error)),
        "candidate_expansion_ratio": float(np.mean(candidate_expansion)),
        "candidate_size": float(np.mean(candidate_sizes)),
        "exact_r25_candidate_size": float(np.mean(exact_r25_sizes)),
        "query_time_ms": float(np.mean(times) * 1000.0),
        "filter_time_ms": float(np.mean(filter_times) * 1000.0),
        "kde_time_ms": float(np.mean(kde_times) * 1000.0),
        "build_time_s": float(getattr(index, "build_time_s", 0.0)),
        "raw_hash_size": float(np.mean(diagnostics["raw_hash_size"])),
        "outer_prefilter_size": float(np.mean(diagnostics["outer_prefilter_size"])),
        "inner_prefilter_size": float(np.mean(diagnostics["inner_prefilter_size"])),
        "outer_hash_size": float(np.mean(diagnostics["outer_hash_size"])),
        "inner_hash_size": float(np.mean(diagnostics["inner_hash_size"])),
        "ring_count": float(np.nanmean(diagnostics["ring_count"])),
        "mean_ring_size": float(np.nanmean(diagnostics["mean_ring_size"])),
        "max_ring_size": float(np.nanmean(diagnostics["max_ring_size"])),
    }


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kde_quality"] = (
        1.0 - df["kde_abs_relative_error_vs_exact_r25"]
    ).clip(lower=0.0, upper=1.0)
    time_max = max(float(df["query_time_ms"].max()), 1e-12)
    df["speed_quality"] = (1.0 - df["query_time_ms"] / time_max).clip(0.0, 1.0)
    df["score_balanced"] = (
        0.35 * df["point_f1"]
        + 0.30 * df["point_recall"]
        + 0.25 * df["kde_quality"]
        + 0.10 * df["speed_quality"]
    )
    df["feasible"] = (
        (df["point_recall"] >= 0.90)
        & (df["point_f1"] >= 0.60)
        & (df["kde_abs_relative_error_vs_exact_r25"] <= 0.10)
    )
    return df


def best_by_dataset_factor(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, factor), part in df.groupby(["dataset", "factor"]):
        feasible = part[part["feasible"]]
        source = feasible if not feasible.empty else part
        best = source.sort_values(
            ["score_balanced", "point_recall", "query_time_ms"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(best)
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

    index_cache: dict[tuple[int, int], exp.MECHHashIndex] = {}

    def get_index(tables: int, bits: int) -> exp.MECHHashIndex:
        key = (int(tables), int(bits))
        if key not in index_cache:
            index_cache[key] = exp.MECHHashIndex(
                model,
                index_x,
                int(tables),
                int(bits),
                int(BASE["hamming_probe"]),
            )
        return index_cache[key]

    rows = []
    sweeps = {
        "L": L_VALUES,
        "K": K_VALUES,
    }
    total = sum(len(values) for values in sweeps.values())
    done = 0
    for factor, values in sweeps.items():
        for value in values:
            tables = int(value) if factor == "L" else int(BASE["L"])
            bits = int(value) if factor == "K" else int(BASE["K"])
            index = get_index(tables, bits)
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
                    "factor": factor,
                    "factor_value": int(value),
                    "L": tables,
                    "K": bits,
                    "hamming_probe": int(BASE["hamming_probe"]),
                    "median_r25": median_r25,
                    "mean_r25": float(np.mean(query_radii)),
                    "mean_bandwidth": float(np.mean(bandwidths)),
                }
            )
            rows.append(row)
            done += 1
            print(
                f"[{dataset} {done:02d}/{total}] {factor}={value}: "
                f"P={row['point_precision']:.3f} R={row['point_recall']:.3f} "
                f"F1={row['point_f1']:.3f} "
                f"KDE={row['kde_abs_relative_error_vs_exact_r25']:.3f} "
                f"C={row['candidate_size']:.1f} T={row['query_time_ms']:.3f}ms",
                flush=True,
            )

    df = score_rows(pd.DataFrame(rows))
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
        "base": BASE,
    }
    return df, meta


def plot_factor_curves(df: pd.DataFrame, factor: str) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
    metrics = [
        ("point_recall", "Recall"),
        ("point_precision", "Precision"),
        ("kde_abs_relative_error_vs_exact_r25", "KDE error"),
        ("candidate_size", "Candidate size"),
        ("query_time_ms", "Query time (ms)"),
    ]
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
    part = df[df["factor"] == factor].copy()
    fig, axes = plt.subplots(1, len(metrics), figsize=(21, 4.2))
    for dataset, current in part.groupby("dataset"):
        current = current.sort_values("factor_value")
        x = current["factor_value"].astype(float).to_numpy()
        for ax, (metric, title) in zip(axes, metrics):
            ax.plot(
                x,
                current[metric].astype(float).to_numpy(),
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                color=colors.get(dataset),
                label=labels.get(dataset, dataset),
            )
    for ax, (_, title) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel(factor)
        ax.grid(True, alpha=0.28)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{factor}_r25_bandwidth014_sensitivity.png", dpi=400)
    fig.savefig(figure_dir / f"{factor}_r25_bandwidth014_sensitivity.pdf")
    plt.close(fig)


def pivot_metric(df: pd.DataFrame, factor: str, metric: str) -> pd.DataFrame:
    return (
        df[df["factor"] == factor]
        .pivot_table(
            index="factor_value",
            columns="dataset",
            values=metric,
            aggfunc="mean",
        )
        .reindex(columns=DATASETS)
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
            if col in {"dataset", "factor"}:
                cells.append(str(value))
            elif col in {"factor_value", "L", "K"}:
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
        result_path = dataset_dir / "r25_lk_sensitivity.csv"
        if result_path.exists():
            print(f"Using existing results for {dataset}", flush=True)
            df = pd.read_csv(result_path)
            meta = {"dataset": dataset, "loaded_existing": True}
        else:
            print(f"=== Dataset: {dataset} ===", flush=True)
            df, meta = evaluate_dataset(dataset)
            df.to_csv(result_path, index=False)
        all_rows.append(df)
        metadata.append(meta)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(OUT_DIR / "r25_lk_sensitivity_all.csv", index=False)
    best = best_by_dataset_factor(results)
    best.to_csv(OUT_DIR / "best_by_dataset_factor.csv", index=False)

    for factor in ["L", "K"]:
        plot_factor_curves(results, factor)
        for metric in [
            "point_recall",
            "point_precision",
            "point_f1",
            "kde_abs_relative_error_vs_exact_r25",
            "candidate_size",
            "query_time_ms",
        ]:
            pivot_metric(results, factor, metric).to_csv(
                OUT_DIR / f"{factor}_{metric}_table.csv",
                index=False,
            )

    config = {
        "datasets": DATASETS,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": RADIUS_PERCENTILE,
        "bandwidth_mode": "h(q)=bandwidth_factor*R25(q)",
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta_mode": "delta=0.25*median_q R25(q)",
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "base": BASE,
        "metrics": {
            "point_recall": "retrieved exact R25 neighbors / exact R25 neighbors",
            "point_precision": "retrieved exact R25 neighbors / retrieved candidates",
            "kde_abs_relative_error_vs_exact_r25": "|KDE_hash - KDE_exact_R25| / KDE_exact_R25",
            "query_time_ms": "median repeated retrieval+KDE time per query",
        },
        "metadata": metadata,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_cols = [
        "dataset",
        "factor",
        "factor_value",
        "L",
        "K",
        "point_precision",
        "point_recall",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "query_time_ms",
        "score_balanced",
        "feasible",
    ]
    summary = [
        "# R25 L/K Sensitivity",
        "",
        "固定查询半径为 query-adaptive R25，核带宽为 h(q)=0.14R25(q)，单独调节哈希表数量 L 和哈希码长度 K。",
        "",
        "## Best Values by Dataset and Factor",
        "",
        markdown_table(best, best_cols),
        "",
        "## Outputs",
        "",
        "- `r25_lk_sensitivity_all.csv`",
        "- `best_by_dataset_factor.csv`",
        "- `figures/L_r25_bandwidth014_sensitivity.png`",
        "- `figures/K_r25_bandwidth014_sensitivity.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved R25 L/K sensitivity results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
