"""Full-target stratified estimator; no access to distances or truth caches.

All H, pilot and main evaluations count against one budget.  IDs supplied in
H and cells must be a disjoint partition of range(n).  Set validate=True for
construction/unit checks; validation scans IDs and must not be hidden online.
"""
import math
import time
import numpy as np


def allocate_counts(sizes, budget, weights=None):
    """Minimum one per nonempty cell, bounded deterministic apportionment.

    With weights=None the remaining budget follows residual capacities.  With
    weights, remaining units follow those nonnegative weights, with saturation
    followed by reapportionment.  Ties use increasing cell index.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    if np.any(sizes < 0):
        raise ValueError('Negative cell size')
    budget = min(int(budget), int(sizes.sum()))
    out = (sizes > 0).astype(np.int64)
    if budget < int(out.sum()):
        raise ValueError('Budget cannot cover every nonempty cell')
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != sizes.shape or np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError('Invalid allocation weights')
    left = budget - int(out.sum())
    while left:
        capacity = sizes - out
        active = capacity > 0
        w = capacity.astype(float) if weights is None else np.where(active, weights, 0.)
        if not float(w.sum()) > 0.:
            w = capacity.astype(float)
        quota = left * w / float(w.sum())
        increment = np.minimum(capacity, np.floor(quota).astype(np.int64))
        used = int(increment.sum())
        if used:
            out += increment
            left -= used
            continue
        # Each active quota is below one. Largest remainder completes this pass.
        order = np.argsort(-quota, kind='stable')
        order = order[active[order]][:left]
        if not len(order):
            raise RuntimeError('Allocation made no progress')
        out[order] += 1
        left -= len(order)
    return out


def _draw_positions(size, count, rng):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == size:
        return np.arange(size, dtype=np.int64)
    return np.asarray(rng.choice(size, size=int(count), replace=False), dtype=np.int64)


def _expand_complement_positions(positions, excluded):
    """Map uniform positions in a compressed complement without an N-size mask."""
    out = np.asarray(positions, dtype=np.int64).copy()
    for position in np.sort(excluded):
        out += out >= position
    return out


def _logsum(values):
    if not len(values):
        return -math.inf
    return float(np.logaddexp.reduce(values))


def _validate_partition(n, H, cells):
    ids = np.concatenate([H] + cells)
    if len(ids) != n or np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != n:
        raise ValueError('H and cells must be a disjoint full-population partition')


def estimate_strata(n, H, cells, budget, rng, kernel_fn, allocation='proportional',
                    pilot_max=2, shrinkage=0.5, validate=False):
    """Estimate n^-1 sum K from exact H plus sampled disjoint full-target cells.

    kernel_fn(ids) returns log K for those identifiers in the ORIGINAL space.
    It is called once in proportional mode, at most twice in Neyman mode.
    """
    clock = time.perf_counter_ns
    begin = clock()
    sampling_ns = allocation_ns = kernel_ns = aggregation_ns = 0
    t = clock()
    n, budget, pilot_max = int(n), int(budget), int(pilot_max)
    if n <= 0 or budget < 0:
        raise ValueError('n must be positive and budget nonnegative')
    if allocation not in ('proportional', 'neyman'):
        raise ValueError('allocation must be proportional or neyman')
    if not 0. <= shrinkage <= 1. or pilot_max < 0 or pilot_max > 2:
        raise ValueError('Require 0<=shrinkage<=1 and 0<=pilot_max<=2')
    H = np.asarray(H, dtype=np.int64).reshape(-1)
    cells = [np.asarray(cell, dtype=np.int64).reshape(-1) for cell in cells]
    input_cell_sizes = np.asarray([len(cell) for cell in cells], dtype=np.int64)
    cells = [cell for cell in cells if len(cell)]
    if len(H) + sum(map(len, cells)) != n:
        raise ValueError('H and cells have incorrect total cardinality')
    if validate:
        _validate_partition(n, H, cells)
    limit = min(budget, n)
    if len(H) > limit:
        raise ValueError('Exact H exceeds total budget')
    available = limit - len(H)
    if cells and available == 0:
        raise ValueError('No budget remains for the unobserved population')
    fallback = ''
    merged = False
    if len(cells) > available:
        # Membership-only decision before any kernel observation.
        groups = np.array_split(np.arange(len(cells)), available)
        cells = [np.concatenate([cells[int(j)] for j in group]) for group in groups]
        fallback = 'adjacent_cells_merged_before_sampling'
        merged = True
    sizes = np.asarray([len(cell) for cell in cells], dtype=np.int64)
    K = len(cells)
    pilot_counts = np.zeros(K, dtype=np.int64)
    main_counts = np.zeros(K, dtype=np.int64)
    effective = allocation
    if available == int(sizes.sum()):
        effective = 'census'
        main_counts = sizes.copy()
    elif allocation == 'neyman':
        proposed = np.minimum(sizes, pilot_max)
        required = int(proposed.sum() + np.count_nonzero(sizes - proposed))
        if pilot_max == 0 or required > available:
            effective = 'proportional'
            fallback = (fallback + ';' if fallback else '') + 'pilot_plus_main_minimum_exceeds_budget'
            main_counts = allocate_counts(sizes, available)
        else:
            pilot_counts = proposed
    else:
        main_counts = allocate_counts(sizes, available)
    allocation_ns += clock() - t

    calls = []
    def evaluate(ids):
        nonlocal kernel_ns
        if not len(ids):
            return np.empty(0, dtype=np.float64)
        start = clock()
        logs = np.asarray(kernel_fn(ids), dtype=np.float64).reshape(-1)
        kernel_ns += clock() - start
        if len(logs) != len(ids) or np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
            raise ValueError('kernel_fn returned invalid log values')
        calls.append(len(ids))
        return logs

    pilot_positions = [np.empty(0, dtype=np.int64) for _ in cells]
    pilot_ids = [np.empty(0, dtype=np.int64) for _ in cells]
    main_ids = [np.empty(0, dtype=np.int64) for _ in cells]
    pilot_logs = [np.empty(0, dtype=np.float64) for _ in cells]
    sigma = np.zeros(K, dtype=np.float64)
    if effective == 'neyman':
        t = clock()
        for j, cell in enumerate(cells):
            pilot_positions[j] = _draw_positions(len(cell), pilot_counts[j], rng)
            pilot_ids[j] = cell[pilot_positions[j]]
        first_ids = np.concatenate([H] + pilot_ids)
        sampling_ns += clock() - t
        first_logs = evaluate(first_ids)
        t = clock()
        offset = len(H)
        for j, count in enumerate(pilot_counts):
            pilot_logs[j] = first_logs[offset:offset + count]
            offset += count
        pilots = first_logs[len(H):]
        # Scale by the largest pilot log value: global scaling cancels in weights.
        finite = pilots[np.isfinite(pilots)]
        scale = float(finite.max()) if len(finite) else 0.
        values = np.exp(pilots - scale)
        pooled_variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.
        for j, logs in enumerate(pilot_logs):
            variance = float(np.var(np.exp(logs - scale), ddof=1)) if len(logs) > 1 else 0.
            sigma[j] = math.sqrt((1. - shrinkage) * variance + shrinkage * pooled_variance)
        remaining_sizes = sizes - pilot_counts
        main_counts = allocate_counts(remaining_sizes, available - int(pilot_counts.sum()),
                                      remaining_sizes * sigma)
        allocation_ns += clock() - t
        t = clock()
        for j, cell in enumerate(cells):
            compressed = _draw_positions(int(remaining_sizes[j]), int(main_counts[j]), rng)
            positions = _expand_complement_positions(compressed, pilot_positions[j])
            main_ids[j] = cell[positions]
        second_ids = np.concatenate(main_ids) if K else np.empty(0, dtype=np.int64)
        sampling_ns += clock() - t
        second_logs = evaluate(second_ids)
        t = clock()
        log_terms = [_logsum(first_logs)]
        offset = 0
        for size, count in zip(remaining_sizes, main_counts):
            if size:
                if count < 1:
                    raise RuntimeError('Unobserved nonempty stratum has no main sample')
                log_terms.append(_logsum(second_logs[offset:offset + count]) + math.log(size / count))
            offset += count
        selected_ids = np.concatenate([first_ids, second_ids])
        selected_logs = np.concatenate([first_logs, second_logs])
    else:
        t = clock()
        for j, cell in enumerate(cells):
            main_ids[j] = cell[_draw_positions(len(cell), main_counts[j], rng)]
        selected_ids = np.concatenate([H] + main_ids)
        sampling_ns += clock() - t
        selected_logs = evaluate(selected_ids)
        t = clock()
        log_terms = [_logsum(selected_logs[:len(H)])]
        offset = len(H)
        for size, count in zip(sizes, main_counts):
            log_terms.append(_logsum(selected_logs[offset:offset + count]) + math.log(size / count))
            offset += count
    log_estimate = _logsum(np.asarray(log_terms, dtype=np.float64)) - math.log(n)
    aggregation_ns += clock() - t
    if len(selected_ids) != limit:
        raise RuntimeError('Estimator did not consume the prescribed capped budget')
    result = {
        'log_estimate': log_estimate, 'actual_samples': len(selected_ids),
        'selected_ids': selected_ids, 'selected_log_kernel_values': selected_logs,
        'H_ids': H, 'H_count': len(H), 'cell_sizes': sizes,
        'input_cell_sizes': input_cell_sizes, 'pilot_counts': pilot_counts,
        'main_counts': main_counts, 'pilot_ids': pilot_ids, 'main_ids': main_ids,
        'allocation': allocation, 'effective_allocation': effective,
        'fallback': fallback, 'merged_cells': merged, 'shrinkage': float(shrinkage),
        'scaled_shrunk_sigma': sigma, 'kernel_calls': len(calls),
        'kernel_call_sizes': calls, 'sampling_ns': sampling_ns,
        'allocation_ns': allocation_ns, 'kernel_ns': kernel_ns,
        'aggregation_ns': aggregation_ns,
    }
    result['total_ns'] = clock() - begin
    result['unassigned_ns'] = result['total_ns'] - sum(result[k] for k in
        ['sampling_ns', 'allocation_ns', 'kernel_ns', 'aggregation_ns'])
    if result['unassigned_ns'] < 0:
        raise RuntimeError('Overlapping estimator profiling phases')
    return result
