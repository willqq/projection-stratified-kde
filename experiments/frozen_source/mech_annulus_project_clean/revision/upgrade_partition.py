"""Exact membership-equivalent replacement for full score sorting.

The cells are returned in score-rank order, but IDs inside a cell need not be
sorted. Uniform sampling without replacement is invariant to this permutation.
Boundary score ties use the original full (score, ID) ordering as a safe exact
fallback. This is an implementation change, not a new statistical method.
"""
import time
import numpy as np


def split_scores(scores, ids, k, groups=8, algorithm='multiselect'):
    """Partition finite scores with unique IDs into H and equal-size cells.

    The caller owns input validity. Avoid an extra full validation scan online;
    the separate correctness suite checks these preconditions and memberships.
    Sizes match np.array_split(remainder, min(groups, len(remainder))).
    """
    begin = time.perf_counter_ns()
    scores = np.asarray(scores)
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    if scores.shape != (n,) or not 0 <= k <= n or groups < 1:
        raise ValueError('Invalid score/ID dimensions, H count or group count')
    remaining = n - k
    number = min(groups, remaining)
    if number:
        sizes = np.full(number, remaining // number, dtype=np.int64)
        sizes[:remaining % number] += 1
        edges = np.r_[k, k + np.cumsum(sizes)]
    else:
        edges = np.array([k], dtype=np.int64)
    cuts = edges[(edges > 0) & (edges < n)]
    layout_ns = time.perf_counter_ns() - begin

    t = time.perf_counter_ns()
    fallback = False
    if algorithm == 'lexsort':
        order = np.lexsort((ids, scores))
    elif algorithm == 'multiselect':
        if len(cuts):
            ranks = np.unique(np.r_[cuts - 1, cuts])
            order = np.argpartition(scores, ranks)
            # Both ranks adjacent to every boundary are fixed order statistics.
            # A cut through ties needs the original deterministic ID convention.
            fallback = bool(np.any(scores[order[cuts - 1]] == scores[order[cuts]]))
            if fallback:
                order = np.lexsort((ids, scores))
        else:
            order = np.arange(n, dtype=np.int64)
    else:
        raise ValueError('Unknown partition implementation')
    ordering_ns = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    ranked = ids[order]
    gather_ns = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    H = ranked[:k]
    cells = [ranked[int(a):int(b)] for a, b in zip(edges[:-1], edges[1:])]
    slicing_ns = time.perf_counter_ns() - t
    return H, cells, dict(
        partition_layout_ns=layout_ns, partition_ordering_ns=ordering_ns,
        partition_id_gather_ns=gather_ns, partition_slicing_ns=slicing_ns,
        partition_total_ns=time.perf_counter_ns() - begin,
        partition_tie_fallback=int(fallback), partition_id_copy_count=n,
        partition_original_vector_copy_count=0)


def profile_original(scores, ids, k, groups=8):
    """Instrument the original expressions, without the new layout machinery."""
    begin = time.perf_counter_ns()
    order = np.lexsort((ids, scores))
    sorting_ns = time.perf_counter_ns() - begin
    t = time.perf_counter_ns()
    ranked = ids[order]
    gather_ns = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    H = ranked[:k]
    cells = [x for x in np.array_split(ranked[k:], min(groups, max(1, len(ids)-k))) if len(x)]
    slicing_ns = time.perf_counter_ns() - t
    return H, cells, dict(original_sort_ns=sorting_ns,
        original_id_gather_ns=gather_ns, original_slicing_ns=slicing_ns,
        original_sorting_and_strata_ns=time.perf_counter_ns()-begin)
