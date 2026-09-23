"""Fixed-candidate mechanism diagnostics with the final frozen partition.

Identical protocol to round-1 mechanism.py (first 20 fixed test rows of
ISOLET/GIST, budgets 64/128/256, 200 repeats, 5 random partitions, identical
seed constants), but the partition under evaluation is the final calibrated
configuration read from the frozen config file.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import core as c
import run as r
import final_run

CFG_DATASETS = ['isolet', 'cifar10_gist512']
QUERIES = 20
BUDGETS = [64, 128, 256]
REPEATS = 200
RANDOM_PARTITIONS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--run-id', default='mechanism_final')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    arrays, split, meta, e, e_base, assets = final_run.build(args.dataset, cfg)
    out = r.JOBROOT / 'results' / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    rawpath = out / (args.dataset + '_mechanism_draws.csv')
    if rawpath.exists():
        raise RuntimeError('Do not overwrite completed or partial results')
    draws, summaries, decompositions, partitions = [], [], [], []
    for qi, q in enumerate(arrays['test'][:QUERIES]):
        union, _, rings = e.partition(q, e.index.query_codes(np.asarray(q, dtype=np.float32)), e.radius)
        C = np.asarray(sorted(union), dtype=int)
        logk = -c.sqdist(e.points, q) / (2 * e.h * e.h)
        shift = float(logk.max())
        values = np.exp(logk - shift)
        full = float(values.sum() / e.n)
        inside = (-2 * e.h * e.h * logk) <= e.radius ** 2
        truncated = float(values[inside].sum() / e.n)
        candidate = float(values[C].sum() / e.n)
        cmask = np.zeros(e.n, dtype=bool); cmask[C] = True
        missing = float(values[inside & ~cmask].sum() / e.n)
        extra = float(values[~inside & cmask].sum() / e.n)
        decompositions.append(dict(dataset=args.dataset, query_id=qi,
            raw_row_id=int(split['test'][qi]), log_kernel_scale=shift, full_scaled=full,
            truncated_scaled=truncated, candidate_scaled=candidate, missing_scaled=missing,
            extra_scaled=extra, tail_scaled=full - truncated,
            signed_candidate_discrepancy_scaled=candidate - truncated,
            candidate_size=len(C)))
        assert np.isclose(candidate - truncated, extra - missing, rtol=1e-10, atol=1e-14)
        for budget in BUDGETS:
            approximate = c.prepare_rings(rings, budget)
            sizes = [len(x) for x in approximate]
            counts = c.allocate(sizes, budget)
            boundaries = np.cumsum(sizes)[:-1]
            ordered = C[np.argsort(logk[C], kind='stable')[::-1]] if len(C) else C
            ordered_cells = list(np.split(ordered, boundaries)) if len(C) else []
            cases = [('approx_annular', 0, approximate, counts),
                     ('exact_distance_order', 0, ordered_cells, counts),
                     ('uniform_on_C', 0, [C] if len(C) else [],
                      np.array([min(budget, len(C))]) if len(C) else np.zeros(0, dtype=int))]
            for ps in range(RANDOM_PARTITIONS):
                shuffled = np.random.default_rng(400000 + qi * 100 + ps).permutation(C)
                cases.append(('random_partition', ps,
                              list(np.split(shuffled, boundaries)) if len(C) else [], counts))
            for label, ps, cells, alloc in cases:
                pred = c.predicted_variance([values[x] for x in cells], alloc, e.n)
                estimates = []
                partitions.append(dict(query_id=qi, budget=budget, grouping=label,
                                       partition_seed=ps,
                                       cell_ids=[x.tolist() for x in cells],
                                       allocation=np.asarray(alloc).tolist()))
                for rep in range(REPEATS):
                    rng = np.random.default_rng(700000 + qi * 100000 + budget * 1000 +
                                                ps * REPEATS + rep)
                    ids, weights = c.sample_ids(cells, alloc, rng)
                    estimate = float(np.sum(values[ids] * np.exp(weights)) / e.n)
                    estimates.append(estimate)
                    draws.append(dict(dataset=args.dataset, query_id=qi, budget=budget,
                                      grouping=label, partition_seed=ps, repeat=rep,
                                      estimate_scaled=estimate,
                                      sampling_error_scaled=estimate - candidate,
                                      log_kernel_scale=shift,
                                      total_error_scaled=estimate - full,
                                      actual_samples=len(ids)))
                    assert np.isclose(estimate - full,
                                      (estimate - candidate) + (candidate - truncated) - (full - truncated))
                empirical = float(np.var(estimates, ddof=1))
                mean_error = float(np.mean(estimates) - candidate)
                se = np.sqrt(pred / REPEATS) if pred else 0.0
                summaries.append(dict(dataset=args.dataset, query_id=qi, budget=budget,
                                      grouping=label, partition_seed=ps,
                                      candidate_size=len(C),
                                      actual_samples=int(np.sum(alloc)),
                                      mean_scaled=float(np.mean(estimates)),
                                      candidate_scaled=candidate,
                                      mean_error_scaled=mean_error,
                                      predicted_variance_scaled=pred,
                                      empirical_variance_scaled=empirical,
                                      mean_z=mean_error / se if se else (0. if abs(mean_error) < 1e-12 else np.inf),
                                      variance_ratio=empirical / pred if pred else np.nan,
                                      log_kernel_scale=shift))
        print(json.dumps({'event': 'mechanism_progress', 'dataset': args.dataset,
                          'query': qi + 1}), flush=True)
    pd.DataFrame(draws).to_csv(rawpath, index=False)
    pd.DataFrame(summaries).to_csv(out / (args.dataset + '_mechanism_summary.csv'), index=False)
    pd.DataFrame(decompositions).to_csv(out / (args.dataset + '_decomposition.csv'), index=False)
    (out / (args.dataset + '_partitions.json')).write_text(json.dumps(partitions))
    (out / (args.dataset + '_mechanism_DONE.json')).write_text(json.dumps(
        {'code_commit': r.git_head(), 'draws': len(draws), 'config': cfg, 'time': __import__('time').time()}))


if __name__ == '__main__':
    main()
