"""Fast evaluator: posting retrieval, optional calibrated T(r), stage timing.

Estimator semantics are identical to core.Evaluator (same allocation, sampling
and log-weighted aggregation); only candidate retrieval changes, plus an
optional per-bound Hamming schedule from calib.fit_calibration.
"""
from __future__ import annotations
import time
import numpy as np
import core as c
import fast_partition as fp
import calib


class FastEvaluator(c.Evaluator):
    def __init__(self, points, h, radius, layers=8, index=None, delta=None,
                 min_collisions=2, t_lookup=None):
        super().__init__(points, h, radius, layers, index, delta, min_collisions)
        self.t_lookup = t_lookup  # None -> original default T(r) schedule

    def partition(self, q, codes, radius, stages=None, counters=None):
        q32 = np.asarray(q, dtype=np.float32)
        q_norm = float(np.linalg.norm(q32))
        bounds = np.linspace(0.0, radius, self.layers + 1)[1:]
        if self.t_lookup is None:
            radii = fp.default_hamming_radii(self.index, q_norm, bounds)
        else:
            radii = calib.calibrated_hamming_radii(q_norm, bounds, self.t_lookup)
        return fp.fast_fixed_radius_partition(
            self.index, codes, q_norm, radius, self.layers, self.norms,
            self.norm_layers, self.sphere_table, self.delta, self.min_collisions,
            hamming_radii=radii, stages=stages, counters=counters)

    def evaluate(self, method, q, budget, rng):
        start = time.perf_counter_ns()
        stages = {'encoding_ns': 0, 'radius_ns': 0, 'candidate_ns': 0,
                  'hash_lookup_ns': 0, 'hit_count_ns': 0, 'sphere_ns': 0,
                  'grouping_ns': 0, 'allocation_ns': 0, 'kernel_ns': 0}
        counters = {'distance_evaluations': 0, 'kernel_evaluations': 0,
                    'hash_id_checks': 0, 'code_distance_checks': 0, 'posting_visits': 0}
        t = time.perf_counter_ns()
        radius = float(self.radius)
        stages['radius_ns'] = time.perf_counter_ns() - t
        if method == 'exact':
            t = time.perf_counter_ns()
            value = c.exact_log_mean(self.points, q, self.h)
            stages['kernel_ns'] = time.perf_counter_ns() - t
            counters.update(distance_evaluations=self.n, kernel_evaluations=self.n)
            sampled, candidate_size, sizes, counts = self.n, self.n, [], []
        elif method == 'uniform_mc':
            raw_cells = [self.id_array]
            sizes, counts, ids, logweights, sampled, candidate_size = \
                self._sample(raw_cells, budget, rng, stages)
            value = self._kernel(q, ids, logweights, stages, counters)
        elif method == 'exact_annular':
            t = time.perf_counter_ns()
            full_distances = c.sqdist(self.points, q)
            labels = np.searchsorted(np.linspace(0, radius, self.layers + 1)[1:] ** 2,
                                     full_distances, side='left')
            raw_cells = [np.flatnonzero(labels == i) for i in range(self.layers + 1)]
            counters['distance_evaluations'] = self.n
            stages['candidate_ns'] = time.perf_counter_ns() - t
            sizes, counts, ids, logweights, sampled, candidate_size = \
                self._sample(raw_cells, budget, rng, stages)
            value = self._kernel(q, ids, logweights, stages, counters, full_distances)
        elif method == 'approx_annular_mech':
            t = time.perf_counter_ns()
            codes = self.index.query_codes(np.asarray(q, dtype=np.float32))
            stages['encoding_ns'] = time.perf_counter_ns() - t
            t = time.perf_counter_ns()
            _, _, raw_cells = self.partition(q, codes, radius, stages, counters)
            stages['candidate_ns'] = time.perf_counter_ns() - t
            sizes, counts, ids, logweights, sampled, candidate_size = \
                self._sample(raw_cells, budget, rng, stages)
            value = self._kernel(q, ids, logweights, stages, counters)
        else:
            raise ValueError(method)
        total = time.perf_counter_ns() - start
        # candidate_ns above measures the whole partition call; the disjoint
        # sub-stages are reported separately, so keep the remainder as residual.
        sub = (stages['hash_lookup_ns'] + stages['hit_count_ns'] +
               stages['sphere_ns'] + stages['grouping_ns'])
        stages['candidate_residual_ns'] = max(stages['candidate_ns'] - sub, 0)
        return dict(log_estimate=value, time_ns=total, actual_samples=sampled,
                    candidate_size=candidate_size, nonempty_layers=len(sizes),
                    layer_sizes=sizes, allocation=np.asarray(counts).tolist(),
                    **stages, **counters)

    def _sample(self, raw_cells, budget, rng, stages):
        t = time.perf_counter_ns()
        cells = c.prepare_rings(raw_cells, budget)
        sizes = [len(x) for x in cells]
        counts = c.allocate(sizes, budget)
        ids, logweights = c.sample_ids(cells, counts, rng)
        sampled, candidate_size = len(ids), sum(sizes)
        stages['allocation_ns'] = time.perf_counter_ns() - t
        return sizes, counts, ids, logweights, sampled, candidate_size

    def _kernel(self, q, ids, logweights, stages, counters, full_distances=None):
        t = time.perf_counter_ns()
        if len(ids) == 0:
            stages['kernel_ns'] = time.perf_counter_ns() - t
            return -float('inf')
        if full_distances is not None:
            distances = full_distances[ids]
        else:
            distances = c.sqdist(self.points[ids], q)
        value = c.log_weighted_estimate(-distances / (2 * self.h * self.h), logweights, self.n)
        counters['kernel_evaluations'] += len(ids)
        counters['distance_evaluations'] += len(ids)
        stages['kernel_ns'] = time.perf_counter_ns() - t
        return value
