"""Posting-list hash retrieval with identical candidate semantics.

Replaces the per-radius full scan (np.isin over all reference codes and the
n-element membership mask) with direct bucket gather + collision counting over
visited postings only. Produces the same sets as hash_filter_candidates and the
same rings as approx_fixed_radius_partition_radius_first (variant 'full'),
verified by test_fast_partition.py. Also accepts an explicit per-bound Hamming
radius schedule for calibrated grouping; the default schedule reproduces the
original T(r)=max(2, ceil(c*Theta/pi)) exactly.
"""
from __future__ import annotations
import math
import time
import numpy as np
import mech_annulus_experiments as source


def posting_hash_ids(index, q_codes: list[int], hamming_radius: int, min_collisions: int,
                     stages: dict | None = None, counters: dict | None = None) -> np.ndarray:
    """ids with >= min_collisions table hits within Hamming radius.

    Cost-aware route: when the Hamming ball covers much of the occupied code
    space (loose radii under the calibrated schedule), a vectorized
    popcount-of-xor test over reference codes is cheaper than gathering
    thousands of posting lists. Both routes return identical sets.
    """
    import calib
    required = min(max(int(min_collisions), 1), len(q_codes))
    t0 = time.perf_counter_ns()
    total_neighbor = 0
    total_occupied = 0
    missing_code = False
    for table_id, q_code in enumerate(q_codes):
        dtab = index.distance_tables[table_id].get(int(q_code))
        if dtab is None:
            # Query code unoccupied: the distance table has no precomputed
            # neighbors; the vectorized route handles this case exactly.
            missing_code = True
            continue
        total_neighbor += sum(len(dtab.get(d, ())) for d in range(int(hamming_radius) + 1))
        total_occupied += len(index.tables[table_id])
    use_vector = missing_code or (total_neighbor > max(64, 0.25 * total_occupied / max(len(q_codes), 1)))
    if use_vector:
        counts = np.zeros(len(index.x), dtype=np.int16)
        for table_id, q_code in enumerate(q_codes):
            codes = index.point_codes[table_id]
            dh = calib._popcount(codes ^ np.uint64(int(q_code)))
            counts += (dh <= hamming_radius).astype(np.int16)
        passed = np.flatnonzero(counts >= required).astype(np.int64)
        visits = int(len(index.x) * len(q_codes))
    else:
        gathered = []
        for table_id, q_code in enumerate(q_codes):
            codes = source.distance_table_neighbor_codes(index, table_id, int(q_code), hamming_radius)
            table = index.tables[table_id]
            for code in codes:
                bucket = table.get(int(code))
                if bucket:
                    gathered.append(bucket)
        if not gathered:
            if stages is not None:
                stages['hash_lookup_ns'] += time.perf_counter_ns() - t0
            if counters is not None:
                counters['posting_visits'] = 0
            return np.zeros(0, dtype=np.int64)
        concat = np.concatenate(gathered) if len(gathered) > 1 else np.asarray(gathered[0], dtype=np.int64)
        visits = int(len(concat))
        ids, counts2 = np.unique(concat, return_counts=True)
        passed = ids[counts2 >= required].astype(np.int64)
    t1 = time.perf_counter_ns()
    if stages is not None:
        stages['hash_lookup_ns'] += t1 - t0
        stages['hit_count_ns'] = 0
    if counters is not None:
        counters['posting_visits'] = visits
    return passed


def default_hamming_radii(index, q_norm: float, bounds: np.ndarray) -> list[int]:
    radii = []
    for b in bounds:
        angle = source.max_angle_from_radius(q_norm, float(b))
        radii.append(int(max(index.hamming_probe,
                             min(index.bits, math.ceil(index.bits * angle / math.pi)))))
    return radii


def fast_fixed_radius_partition(
    index,
    q_codes: list[int],
    q_norm: float,
    radius: float,
    annulus_count: int,
    norms: np.ndarray,
    layers: np.ndarray,
    sphere_table: dict[int, np.ndarray],
    delta: float,
    min_collisions: int,
    hamming_radii: list[int] | None = None,
    stages: dict | None = None,
    counters: dict | None = None,
):
    """Same rings/candidate union as source.approx_fixed_radius_partition_radius_first.

    stages accumulates hash_lookup_ns, hit_count_ns, sphere_ns, grouping_ns.
    """
    if stages is None:
        stages = {}
    stages.setdefault('hash_lookup_ns', 0)
    stages.setdefault('hit_count_ns', 0)
    stages.setdefault('sphere_ns', 0)
    stages.setdefault('grouping_ns', 0)
    if counters is None:
        counters = {}
    annulus_count = max(int(annulus_count), 1)
    bounds = np.linspace(0.0, radius, annulus_count + 1)[1:]
    if hamming_radii is None:
        hamming_radii = default_hamming_radii(index, q_norm, bounds)

    t0 = time.perf_counter_ns()
    cache: dict[int, np.ndarray] = {}

    def hash_ids_for(T: int) -> np.ndarray:
        if T not in cache:
            cache[T] = posting_hash_ids(index, q_codes, T, min_collisions, stages, counters)
        return cache[T]

    zero_ids = getattr(index, 'zero_norm_ids', None)
    if zero_ids is None:
        zero_ids = set(np.flatnonzero(norms == 0).tolist())
    zero_arr = np.asarray(sorted(zero_ids), dtype=np.int64) if zero_ids else np.zeros(0, dtype=np.int64)

    # Outermost sphere band over the full radius, reused for every annulus.
    low_full, high_full = source.sphere_layer_bounds(q_norm, radius, delta)
    rows = [sphere_table[layer] for layer in range(low_full, high_full + 1) if layer in sphere_table]
    final_ids = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
    final_ids = np.unique(final_ids) if len(final_ids) else final_ids
    stages['sphere_ns'] += time.perf_counter_ns() - t0

    t1 = time.perf_counter_ns()
    hash_before_loop = stages['hash_lookup_ns']
    previous: np.ndarray | None = None
    ring_candidates: list[set[int]] = []
    partition_candidates: set[int] = set()
    ring_sizes = []
    final_hash = np.zeros(0, dtype=np.int64)
    last_inner_hash = np.zeros(0, dtype=np.int64)
    last_inner_pool_size = 0
    final_pool_size = int(len(final_ids))
    raw_hash = hash_ids_for(int(hamming_radii[-1]))
    for j, outer_radius in enumerate(bounds):
        low, high = source.sphere_layer_bounds(q_norm, float(outer_radius), delta)
        if len(final_ids):
            current_ids = final_ids[(layers[final_ids] >= low) & (layers[final_ids] <= high)]
        else:
            current_ids = final_ids
        T = int(hamming_radii[j])
        passed = hash_ids_for(T)
        # Zero-norm references carry no angle; their radial rule is preserved.
        keep = np.isin(current_ids, passed, assume_unique=False)
        current_passed = current_ids[keep]
        if len(current_passed):
            current_passed = current_passed[~np.isin(current_passed, zero_arr)]
        if q_norm <= outer_radius and len(zero_arr):
            current_passed = np.union1d(current_passed, zero_arr)
        if previous is None:
            ring = current_passed
        else:
            ring = np.setdiff1d(current_passed, previous, assume_unique=True)
        partition_candidates.update(ring.tolist())
        ring_candidates.append(set(ring.tolist()))
        ring_sizes.append(len(ring))
        last_inner_pool_size = len(current_ids) if j + 1 < len(bounds) else last_inner_pool_size
        last_inner_hash = current_passed if j + 1 < len(bounds) else last_inner_hash
        previous = current_passed
        final_pool_size = len(current_ids)
        final_hash = current_passed
    loop_wall = time.perf_counter_ns() - t1
    stages['grouping_ns'] += max(loop_wall - (stages['hash_lookup_ns'] - hash_before_loop), 0)
    stats = {
        'raw_hash_size': float(len(raw_hash)),
        'outer_prefilter_size': float(final_pool_size),
        'inner_prefilter_size': float(last_inner_pool_size),
        'outer_hash_size': float(len(final_hash)),
        'inner_hash_size': float(len(last_inner_hash)),
        'ring_count': float(annulus_count),
        'mean_ring_size': float(np.mean(ring_sizes)) if ring_sizes else 0.0,
        'max_ring_size': float(np.max(ring_sizes)) if ring_sizes else 0.0,
    }
    return partition_candidates, stats, ring_candidates
