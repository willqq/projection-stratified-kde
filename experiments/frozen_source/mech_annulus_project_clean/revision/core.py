"""Audited adapters for a common full-reference Gaussian kernel mean.

The existing Full MECH partition is reused. Evaluation receives no truth/distance
cache. New protocol: train-only preprocessing, validation-fixed R and h, fixed
positive without-replacement allocations, and complete per-query timing.
"""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mech_annulus_experiments as source


def fit_transform_splits(raw, split):
    scaler = MinMaxScaler().fit(raw[split['train']])
    training = scaler.transform(raw[split['train']]).astype(np.float32)
    anchor = training[np.argmax(np.linalg.norm(training, axis=1))].copy()
    arrays = {k: (2 * scaler.transform(raw[v]) + anchor).astype(np.float32)
              for k, v in split.items()}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite input after preprocessing')
    return arrays, scaler, anchor


def sqdist(points, q):
    return cdist(np.asarray(q, dtype=np.float64)[None, :], points,
                 metric='sqeuclidean')[0]


def exact_log_mean(points, q, h):
    return float(logsumexp(-sqdist(points, q) / (2*h*h)) - math.log(len(points)))


def prepare_rings(rings, budget):
    if budget < 1:
        raise ValueError('budget must be positive')
    cells = [np.asarray(sorted(r) if isinstance(r, set) else r, dtype=np.int64) for r in rings if len(r)]
    # Deterministic merge of the two innermost remaining cells when necessary.
    while len(cells) > budget:
        cells[:2] = [np.concatenate(cells[:2])]
    return cells


def allocate(sizes, budget):
    sizes = np.asarray(sizes, dtype=np.int64)
    if len(sizes) == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(sizes < 1) or budget < len(sizes):
        raise ValueError('Merge first; each nonempty cell needs a sample')
    target = min(int(budget), int(sizes.sum()))
    counts = np.ones(len(sizes), dtype=np.int64)
    while counts.sum() < target:
        remaining = target - int(counts.sum())
        capacity = sizes - counts
        desired = remaining * capacity / capacity.sum()
        increment = np.minimum(np.floor(desired).astype(np.int64), capacity)
        if increment.sum() == 0:
            order = np.argsort(-desired, kind='stable')
            increment[order[:remaining]] = 1
        counts += increment
    return counts


def sample_ids(cells, counts, rng):
    ids, logweights = [], []
    for cell, m in zip(cells, counts):
        chosen = cell if m == len(cell) else rng.choice(cell, int(m), replace=False)
        ids.append(chosen)
        logweights.append(np.full(len(chosen), math.log(len(cell)/m)))
    if not ids:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(ids), np.concatenate(logweights)


def log_weighted_estimate(logk, logweights, n):
    if len(logk) == 0:
        return -math.inf
    return float(logsumexp(logk + logweights) - math.log(n))


def errors(logestimate, logtruth):
    if not math.isfinite(logtruth):
        raise ValueError('Finite positive Gaussian truth required in log space')
    rel = -1.0 if logestimate == -math.inf else float(np.expm1(logestimate-logtruth))
    logabs = -math.inf if rel == 0 else logtruth + math.log(abs(rel))
    return {'truth': float(np.exp(logtruth)), 'log_truth': logtruth,
            'estimate': float(np.exp(logestimate)), 'log_estimate': logestimate,
            'signed_relative_error': rel, 'relative_error': abs(rel),
            'absolute_error': float(np.exp(logabs)), 'log_absolute_error': logabs,
            'zero_estimate': int(logestimate == -math.inf),
            'truth_float_underflow': int(np.exp(logtruth) == 0)}


class Evaluator:
    def __init__(self, points, h, radius, layers=8, index=None, delta=None,
                 min_collisions=2):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.n = len(points)
        self.h, self.radius, self.layers = float(h), float(radius), int(layers)
        self.index, self.delta = index, delta
        self.min_collisions = min_collisions
        if h <= 0 or radius <= 0 or self.n == 0:
            raise ValueError('Positive h/R and nonempty reference required')
        # Offline reference metadata; never contains query distances.
        self.norms = np.linalg.norm(np.asarray(points, dtype=np.float32), axis=1)
        self.all_ids = set(range(self.n))
        self.id_array = np.arange(self.n, dtype=np.int64)
        if index is not None:
            self.norm_layers = np.floor(self.norms/delta).astype(int)
            self.sphere_table = source.build_sphere_table(self.norms, delta)
            self.index.zero_norm_ids = set(np.flatnonzero(self.norms == 0).tolist())

    def partition(self, q, codes, radius):
        self.index.query_stats = {'hash_id_checks': 0, 'code_distance_checks': 0}
        return source.approx_fixed_radius_partition_radius_first(
            self.index, codes, np.asarray(q, dtype=np.float32),
            float(np.linalg.norm(np.asarray(q, dtype=np.float32))), radius,
            self.layers, self.norms, self.norm_layers, self.sphere_table,
            self.delta, 'full', self.min_collisions, self.all_ids)

    def evaluate(self, method, q, budget, rng):
        start = time.perf_counter_ns()
        stages = {'encoding_ns': 0, 'radius_ns': 0, 'candidate_ns': 0,
                  'allocation_ns': 0, 'kernel_ns': 0}
        counters = {'distance_evaluations': 0, 'kernel_evaluations': 0,
                    'hash_id_checks': 0, 'code_distance_checks': 0}
        t = time.perf_counter_ns()
        radius = float(self.radius)  # validation-fixed rule, no test distances
        stages['radius_ns'] = time.perf_counter_ns()-t
        full_distances = None
        if method == 'exact':
            t = time.perf_counter_ns()
            value = exact_log_mean(self.points, q, self.h)
            stages['kernel_ns'] = time.perf_counter_ns()-t
            counters.update(distance_evaluations=self.n, kernel_evaluations=self.n)
            sampled, candidate_size, sizes, counts = self.n, self.n, [], []
        else:
            if method == 'uniform_mc':
                raw_cells = [self.id_array]
            elif method == 'exact_annular':
                t = time.perf_counter_ns()
                full_distances = sqdist(self.points, q)
                labels = np.searchsorted(np.linspace(0,radius,self.layers+1)[1:]**2,
                                         full_distances, side='left')
                # Last stratum is the tail, so this baseline covers all P.
                raw_cells = [np.flatnonzero(labels == i) for i in range(self.layers+1)]
                counters['distance_evaluations'] = self.n
                stages['candidate_ns'] = time.perf_counter_ns()-t
            elif method == 'approx_annular_mech':
                t = time.perf_counter_ns()
                codes = self.index.query_codes(q)
                stages['encoding_ns'] = time.perf_counter_ns()-t
                t = time.perf_counter_ns()
                _, _, raw_cells = self.partition(q, codes, radius)
                stages['candidate_ns'] = time.perf_counter_ns()-t
                counters.update(self.index.query_stats)
            else:
                raise ValueError(method)
            t = time.perf_counter_ns()
            cells = prepare_rings(raw_cells, budget)
            sizes = [len(c) for c in cells]
            counts = allocate(sizes, budget)
            ids, logweights = sample_ids(cells, counts, rng)
            sampled, candidate_size = len(ids), sum(sizes)
            stages['allocation_ns'] = time.perf_counter_ns()-t
            t = time.perf_counter_ns()
            if full_distances is None:
                distances = sqdist(self.points[ids], q) if sampled else np.zeros(0)
                counters['distance_evaluations'] += sampled
            else:
                distances = full_distances[ids]
            value = log_weighted_estimate(-distances/(2*self.h*self.h), logweights, self.n)
            counters['kernel_evaluations'] = sampled
            stages['kernel_ns'] = time.perf_counter_ns()-t
        total = time.perf_counter_ns()-start
        return dict(log_estimate=value, time_ns=total, actual_samples=sampled,
                    candidate_size=candidate_size, nonempty_layers=len(sizes),
                    layer_sizes=sizes, allocation=np.asarray(counts).tolist(),
                    **stages, **counters)


def predicted_variance(values_by_cell, counts, n):
    total = 0.0
    for values, m in zip(values_by_cell, counts):
        N = len(values)
        if N > 1 and m < N:
            total += N*N*(1-m/N)*float(np.var(values, ddof=1))/m
    return total/(n*n)


def aggregate(frame):
    return frame.groupby(['dataset','method','budget'], as_index=False).agg(
        relative_error=('relative_error','mean'),
        absolute_error=('absolute_error','mean'),
        time_mean_ms=('time_ns', lambda x: x.mean()/1e6),
        time_median_ms=('time_ns', lambda x: x.median()/1e6),
        time_p95_ms=('time_ns', lambda x: x.quantile(.95)/1e6),
        kernel_evaluations=('kernel_evaluations','mean'),
        zero_estimates=('zero_estimate','sum'), observations=('query_id','size'))
