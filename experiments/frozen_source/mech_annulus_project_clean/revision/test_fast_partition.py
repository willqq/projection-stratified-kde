"""Verify the posting-list partition reproduces the original candidate semantics."""
from __future__ import annotations
import sys
import numpy as np
import core as c
import run as r
import fast_partition as fp
import mech_annulus_experiments as source


def check_dataset(name, nqueries=6):
    arrays, split, meta, e, _ = r.setup(name)
    rng = np.random.default_rng(11)
    checked = 0
    for qi, q in enumerate(arrays['validation'][:nqueries]):
        q32 = np.asarray(q, dtype=np.float32)
        q_norm = float(np.linalg.norm(q32))
        codes = e.index.query_codes(q)
        for radius in [e.radius * f for f in (0.25, 0.5, 1.0, 1.5)]:
            for J in (1, 8):
                old_union, old_stats, old_rings = source.approx_fixed_radius_partition_radius_first(
                    e.index, codes, q32, q_norm, radius, J, e.norms, e.norm_layers,
                    e.sphere_table, e.delta, 'full', e.min_collisions, e.all_ids)
                new_union, new_stats, new_rings = fp.fast_fixed_radius_partition(
                    e.index, codes, q_norm, radius, J, e.norms, e.norm_layers,
                    e.sphere_table, e.delta, e.min_collisions)
                assert old_union == new_union, (name, qi, radius, J, len(old_union), len(new_union))
                assert all(a == b for a, b in zip(old_rings, new_rings)), (name, qi, radius, J)
                for key in ('outer_prefilter_size', 'inner_prefilter_size', 'outer_hash_size',
                            'inner_hash_size', 'mean_ring_size', 'max_ring_size'):
                    assert abs(old_stats[key] - new_stats[key]) < 1e-9, (name, key)
                checked += 1
    # Edge cases on synthetic index inputs: absent codes, full radius, tiny radius.
    q = arrays['validation'][0]
    q32 = np.asarray(q, dtype=np.float32)
    q_norm = float(np.linalg.norm(q32))
    codes = e.index.query_codes(q)
    for radius in (1e-9, float(q_norm) * 1.5):
        old = source.approx_fixed_radius_partition_radius_first(
            e.index, codes, q32, q_norm, radius, 8, e.norms, e.norm_layers,
            e.sphere_table, e.delta, 'full', e.min_collisions, e.all_ids)
        new = fp.fast_fixed_radius_partition(
            e.index, codes, q_norm, radius, 8, e.norms, e.norm_layers,
            e.sphere_table, e.delta, e.min_collisions)
        assert old[0] == new[0], ('edge', name, radius)
    print(f'OK {name}: {checked + 2} configurations identical', flush=True)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['isolet']:
        check_dataset(name)
