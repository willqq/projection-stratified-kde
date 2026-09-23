"""Portable interface to the frozen projection and sampling implementation.

Inputs are already preprocessed original-space features. This wrapper removes
the legacy MECH loader from the projection-only path. It does not implement
DEANN or replace the historical timed experiment runner.
"""
import time
import numpy as np
from scipy.spatial.distance import cdist
from ._frozen.sprint_projection import ProjectionScore
from ._frozen.sprint_estimator import estimate_strata


class ProjectedKDE:
    def __init__(self, reference, training, bandwidth):
        self.reference32 = np.asarray(reference, dtype=np.float32)
        self.points = np.asarray(self.reference32, dtype=np.float64)
        self.projection = ProjectionScore(self.reference32, training)
        self.n = len(self.points)
        self.h = float(bandwidth)
        if not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("Positive finite bandwidth required")
        self.ids = np.arange(self.n, dtype=np.int64)

    def partition(self, query, budget, single=False):
        if int(budget) != budget or budget < 1:
            raise ValueError("Budget must be a positive integer")
        score = self.projection.query(query)
        ranked = self.ids[np.lexsort((self.ids, score['d2hat']))]
        k = min(int(budget * .25), self.n, max(0, budget - 1))
        H = ranked[:k]
        if single:
            cells = [ranked[k:]] if k < self.n else []
        else:
            cells = [x for x in np.array_split(ranked[k:], min(8, max(1, self.n-k))) if len(x)]
        return H, cells

    def query(self, query, budget=128, seed=0, single=False):
        """Estimate the full Gaussian kernel mean; return audit IDs and counts.

        Single uses identical H and one uniform remainder. Local wrapper timing
        must not be substituted for the manuscript's historical complete latency.
        """
        start = time.perf_counter_ns()
        q = np.asarray(query, dtype=np.float32)
        H, cells = self.partition(q, budget, single)
        def log_kernel(ids):
            return -cdist(np.asarray(q, dtype=np.float64)[None, :],
                          self.points[ids], metric='sqeuclidean')[0] / (2*self.h*self.h)
        result = estimate_strata(self.n, H, cells, budget,
                                 np.random.default_rng(seed), log_kernel,
                                 allocation='proportional')
        result['estimate'] = float(np.exp(result['log_estimate']))
        result['wrapper_time_ns'] = time.perf_counter_ns() - start
        return result
