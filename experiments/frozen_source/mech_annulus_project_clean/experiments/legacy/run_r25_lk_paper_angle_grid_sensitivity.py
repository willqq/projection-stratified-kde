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
    ANGLE_RECALL_ALPHA,
    BANDWIDTH_FACTOR,
    DELTA_FACTOR,
    K_VALUES,
    NormTable,
    PrefixDistanceLookup,
    build_paper_sphere_table,
    encode_tables,
    hamming_threshold_from_angle,
    layer_angle_from_radius,
    paper_layer_bounds,
    query_adaptive_r25,
)


DATASETS = ["isolet", "cifar10", "cifar10_gist512", "amazon"]
OUT_DIR = Path("r25_lk_paper_angle_grid_sensitivity")
L_VALUES = [5, 10, 15, 20]
MAX_L = max(L_VALUES)
MAX_K = max(K_VALUES)
PREVIOUS_L20_DIR = Path("r25_k10_200_l20_paper_angle_lookup")
NORM_FILTER_RULE = "| ||x|| - ||q|| | <= r(q)"
LOOKUP_RULE = (
    "distance-table hash-bucket lookup, union over L tables, "
    "intersect sphere layers, then exact norm-table filter"
)


def expected_pairs() -> set[tuple[int, int]]:
    return {(int(l_value), int(k_value)) for l_value in L_VALUES for k_value in K_VALUES}


def load_complete_result(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "norm_filter_rule" not in df.columns or not (df["norm_filter_rule"] == NORM_FILTER_RULE).all():
        return None
    if {"L", "K"}.issubset(df.columns):
        pairs = {(int(row.L), int(row.K)) for row in df.itertuples()}
        if expected_pairs().issubset(pairs):
            return df
    return None


def load_seed_rows(dataset: str, result_path: Path) -> pd.DataFrame:
    if result_path.exists():
        df = pd.read_csv(result_path)
        if "norm_filter_rule" in df.columns and (df["norm_filter_rule"] == NORM_FILTER_RULE).all():
            return df
        return pd.DataFrame()

    previous_path = PREVIOUS_L20_DIR / dataset / "K10_200_L20_paper_angle_lookup.csv"
    if not previous_path.exists() or 20 not in L_VALUES:
        return pd.DataFrame()

    df = pd.read_csv(previous_path)
    if "norm_filter_rule" not in df.columns or not (df["norm_filter_rule"] == NORM_FILTER_RULE).all():
        return pd.DataFrame()
    df = df[(df["L"] == 20) & (df["K"].isin(K_VALUES))].copy()
    return df


def build_prefix_lookups(index_bits: list[np.ndarray], bits: int) -> list[PrefixDistanceLookup]:
    return [PrefixDistanceLookup(table_bits, bits) for table_bits in index_bits]


def paper_ball_query_lk(
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
    table_count: int,
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
            table_count,
            ANGLE_RECALL_ALPHA,
        )
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
        selected |= layer_hash_selected & layer_masks[layer] & norm_mask

    return (
        np.flatnonzero(selected).astype(np.int32),
        float(np.mean(thresholds)) if thresholds else 0.0,
        float(np.mean(theta_values)) if theta_values else 0.0,
    )


def evaluate_setting(
    dataset: str,
    table_count: int,
    code_bits: int,
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
    norm_table: NormTable,
    query_norms: np.ndarray,
    build_time_s: float,
) -> dict[str, float | int | str]:
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
        approx_ids, mean_threshold, mean_theta = paper_ball_query_lk(
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
            table_count,
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
        "L": int(table_count),
        "K": int(code_bits),
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": 25.0,
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta": float(delta),
        "angle_alpha": ANGLE_RECALL_ALPHA,
        "sphere_layer_rule": "round(norm/delta)",
        "theta_rule": "layer-wise cosine theorem",
        "norm_filter_rule": NORM_FILTER_RULE,
        "hamming_threshold_rule": "per-query, per-layer Eq. I_threshold with Binomial(K, theta/pi)",
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


def save_dataset_rows(path: Path, rows: pd.DataFrame) -> None:
    rows = rows.sort_values(["dataset", "L", "K"]).drop_duplicates(
        ["dataset", "L", "K"],
        keep="last",
    )
    rows.to_csv(path, index=False)


def evaluate_dataset(dataset: str, result_path: Path) -> tuple[pd.DataFrame, dict]:
    existing = load_seed_rows(dataset, result_path)
    completed = set()
    if not existing.empty:
        completed = {(int(row.L), int(row.K)) for row in existing.itertuples()}

    missing = sorted(expected_pairs() - completed)
    if not missing:
        return existing, {"dataset": dataset, "loaded_existing": True}

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
    norm_table = NormTable.from_norms(norms)
    sphere_table, _ = build_paper_sphere_table(norms, delta)
    layer_masks = {}
    for layer, ids in sphere_table.items():
        mask = np.zeros(len(index_x), dtype=bool)
        mask[ids] = True
        layer_masks[layer] = mask
    query_norms = np.linalg.norm(queries, axis=1)

    rows = existing.to_dict("records") if not existing.empty else []
    total = len(expected_pairs())
    done = len(completed)
    for code_bits in K_VALUES:
        lookups = build_prefix_lookups(index_bits, int(code_bits))
        for table_count in L_VALUES:
            pair = (int(table_count), int(code_bits))
            if pair in completed:
                continue
            row = evaluate_setting(
                dataset,
                int(table_count),
                int(code_bits),
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
                norm_table,
                query_norms,
                build_time_s,
            )
            rows.append(row)
            completed.add(pair)
            done += 1
            save_dataset_rows(result_path, pd.DataFrame(rows))
            print(
                f"[{dataset} {done:03d}/{total}] L={table_count} K={code_bits}: "
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
        "radius_percentile": 25.0,
        "median_r25": median_r25,
        "mean_r25": float(np.mean(query_radii)),
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "mean_bandwidth": float(np.mean(bandwidths)),
        "delta": delta,
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "angle_alpha": ANGLE_RECALL_ALPHA,
        "norm_filter_rule": NORM_FILTER_RULE,
        "lookup_rule": LOOKUP_RULE,
        "train_time_s": train_time_s,
        "encode_time_s": encode_time_s,
        "seeded_from_previous_l20": bool(not existing.empty and 20 in L_VALUES),
    }
    return pd.DataFrame(rows), meta


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


def summarize_common(df: pd.DataFrame) -> pd.DataFrame:
    common = (
        df.groupby(["L", "K"])
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
            mean_score_balanced=("score_balanced", "mean"),
            mean_score_accuracy=("score_accuracy", "mean"),
        )
        .reset_index()
    )
    common["feasible_all"] = (common["min_recall"] >= 0.90) & (
        common["max_kde_error"] <= 0.10
    )
    return common


def best_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in df.groupby("dataset"):
        feasible = part[part["feasible"]]
        source = feasible if not feasible.empty else part

        balanced = source.sort_values(
            ["score_balanced", "point_recall", "kde_quality", "query_time_ms"],
            ascending=[False, False, False, True],
        ).iloc[0].copy()
        balanced["selection"] = "balanced"
        rows.append(balanced)

        accuracy = source.sort_values(
            ["score_accuracy", "point_recall", "kde_quality", "query_time_ms"],
            ascending=[False, False, False, True],
        ).iloc[0].copy()
        accuracy["selection"] = "accuracy"
        rows.append(accuracy)

    return pd.DataFrame(rows)


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.pivot_table(index=["dataset", "K"], columns="L", values=metric, aggfunc="mean")
        .reset_index()
    )


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
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 8.4))
    image = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(L_VALUES)))
    ax.set_xticklabels(L_VALUES)
    ax.set_yticks(range(len(K_VALUES)))
    ax.set_yticklabels(K_VALUES)
    ax.set_xlabel("Hash tables L")
    ax.set_ylabel("Hash code length K")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{filename}.png", dpi=450)
    fig.savefig(figure_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_all_heatmaps(df: pd.DataFrame) -> None:
    specs = [
        ("point_recall", "Recall", "recall_heatmap"),
        ("point_f1", "F1", "f1_heatmap"),
        ("kde_abs_relative_error_vs_exact_r25", "KDE error", "kde_error_heatmap"),
        ("candidate_size", "Candidate size", "candidate_size_heatmap"),
        ("mean_hamming_threshold", "Mean Hamming threshold", "hamming_threshold_heatmap"),
        ("query_time_ms", "Query time (ms)", "query_time_heatmap"),
        ("score_balanced", "Balanced score", "balanced_score_heatmap"),
    ]
    for dataset in DATASETS:
        for metric, title, filename in specs:
            plot_heatmap(df, dataset, metric, f"{dataset}: {title}", filename)


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
            elif col in {"feasible", "feasible_all"}:
                cells.append(str(bool(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(results: pd.DataFrame, metadata: list[dict]) -> None:
    results = score_grid(results)
    results.to_csv(OUT_DIR / "r25_lk_paper_angle_grid_all.csv", index=False)

    best = best_by_dataset(results)
    best.to_csv(OUT_DIR / "best_lk_by_dataset.csv", index=False)

    common = summarize_common(results)
    common.to_csv(OUT_DIR / "common_lk_paper_angle_grid_summary.csv", index=False)

    for metric in [
        "point_recall",
        "point_precision",
        "point_f1",
        "kde_abs_relative_error_vs_exact_r25",
        "candidate_size",
        "mean_hamming_threshold",
        "query_time_ms",
        "score_balanced",
    ]:
        pivot_metric(results, metric).to_csv(OUT_DIR / f"grid_{metric}.csv", index=False)

    plot_all_heatmaps(results)

    top_common = common.sort_values(
        ["mean_score_balanced", "mean_recall", "mean_kde_error", "mean_query_time_ms"],
        ascending=[False, False, True, True],
    ).head(10)
    feasible_common = common[common["feasible_all"]].sort_values(
        ["mean_score_balanced", "mean_recall", "mean_kde_error", "mean_query_time_ms"],
        ascending=[False, False, True, True],
    ).head(10)

    config = {
        "datasets": DATASETS,
        "radius_mode": "query_adaptive_R25",
        "radius_percentile": 25.0,
        "bandwidth_mode": "h(q)=0.14R25(q)",
        "bandwidth_factor": BANDWIDTH_FACTOR,
        "delta_mode": "delta=0.25*median_q R25(q)",
        "L_values": L_VALUES,
        "K_values": K_VALUES,
        "angle_recall_alpha": ANGLE_RECALL_ALPHA,
        "sphere_layer_rule": "s=round(||p||/delta), fs=round((||q||-r)/delta), ls=round((||q||+r)/delta)",
        "theta_rule": "theta=arccos((||q||^2+||p_layer||^2-r^2)/(2||q||||p_layer||))",
        "norm_filter_rule": NORM_FILTER_RULE,
        "hamming_threshold_rule": "For each query q and sphere layer s, compute theta(q,s), then I(q,s)=min I such that BinomialCDF(I; K, theta(q,s)/pi)>1-(1-alpha)^(1/L). Reported mean_hamming_threshold is only an aggregate diagnostic.",
        "lookup_rule": "retrieve bucket codes from distance tables within Hamming distance I, merge hash-table buckets, intersect sphere layers, and apply exact norm-table filter",
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
        "mean_hamming_threshold",
        "query_time_ms",
        "score_balanced",
        "score_accuracy",
        "feasible",
    ]
    common_cols = [
        "L",
        "K",
        "mean_precision",
        "mean_recall",
        "min_recall",
        "mean_f1",
        "mean_kde_error",
        "max_kde_error",
        "mean_candidate_size",
        "mean_hamming_threshold",
        "mean_query_time_ms",
        "mean_score_balanced",
        "feasible_all",
    ]

    summary = [
        "# R25 L/K Paper Angle-Lookup Grid Sensitivity",
        "",
        "Query radius is query-adaptive R25 and bandwidth is h(q)=0.14R25(q).",
        "The grid varies hash tables L and hash code length K under the paper angle-to-Hamming threshold rule.",
        "",
        "## Setup",
        "",
        f"- L grid: {', '.join(str(v) for v in L_VALUES)}.",
        f"- K grid: {', '.join(str(v) for v in K_VALUES)}.",
        "- Hamming threshold: for each query q and sphere layer s, compute theta(q,s), then convert it to I(q,s) by Eq. I_threshold with Binomial(K, theta(q,s)/pi), alpha=0.95.",
        f"- Norm-table filter: {NORM_FILTER_RULE}.",
        "- Feasible criterion: recall >= 0.90 and KDE error <= 0.10.",
        "",
        "## Best L/K by Dataset",
        "",
        markdown_table(best, best_cols),
        "",
        "## Top Common Settings by Balanced Score",
        "",
        markdown_table(top_common, common_cols),
        "",
        "## Top Common Feasible Settings",
        "",
        markdown_table(feasible_common, common_cols),
        "",
        "## Outputs",
        "",
        "- `r25_lk_paper_angle_grid_all.csv`",
        "- `common_lk_paper_angle_grid_summary.csv`",
        "- `best_lk_by_dataset.csv`",
        "- `figures/<dataset>/recall_heatmap.png`",
        "- `figures/<dataset>/kde_error_heatmap.png`",
        "- `figures/<dataset>/balanced_score_heatmap.png`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metadata = []

    for dataset in DATASETS:
        dataset_dir = OUT_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result_path = dataset_dir / "r25_lk_paper_angle_grid.csv"
        complete = load_complete_result(result_path)
        if complete is not None:
            print(f"Using existing L/K paper angle grid results for {dataset}", flush=True)
            df = complete
            meta = {"dataset": dataset, "loaded_existing": True}
        else:
            print(f"=== Dataset: {dataset} ===", flush=True)
            df, meta = evaluate_dataset(dataset, result_path)
        all_rows.append(df)
        metadata.append(meta)

    write_outputs(pd.concat(all_rows, ignore_index=True), metadata)
    print(f"Saved R25 L/K paper angle grid sensitivity to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
