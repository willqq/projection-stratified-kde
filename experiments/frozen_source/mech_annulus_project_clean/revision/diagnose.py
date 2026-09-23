"""Failure diagnosis for the approximate annular estimator (validation queries only).

Answers, per dataset:
  A. candidate-union coverage: id recall / kernel-mass recall / discrepancy / sizes
  B. partition quality: per-ring kernel variance, fixed-C predicted + empirical
     variance ratios (approx / random / ordered / exact-distance strata on C),
     and full-target comparison (uniform MC over P vs exact strata vs approx)
  C. geometry: distinct T values used, norm-band occupancy, angle-Hamming
     calibration of learned MECH codes vs the binomial model
No test query is read. Writes results/diag_v1/.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import core as c
import run as r

BUDGETS = [64, 128, 256]
EMP_REPEATS = 200
CAL_PAIRS = 2048


def ring_stats(rings, values):
    rows = []
    for j, ring in enumerate(rings):
        if not ring:
            rows.append(dict(ring=j, size=0, mean=0.0, var=0.0, cv=0.0, mass=0.0))
            continue
        v = values[np.fromiter(ring, dtype=int)]
        rows.append(dict(ring=j, size=len(ring), mean=float(v.mean()),
                         var=float(v.var(ddof=1)) if len(v) > 1 else 0.0,
                         cv=float(v.std() / max(v.mean(), 1e-300)),
                         mass=float(v.sum())))
    return rows


def empirical_variance(cells, counts, values, n, repeats, rng_seed_base):
    ests = []
    for rep in range(repeats):
        rng = np.random.default_rng(rng_seed_base + rep)
        ids, lw = c.sample_ids(cells, counts, rng)
        ests.append(float(np.sum(values[ids] * np.exp(lw)) / n))
    return float(np.var(ests, ddof=1)), float(np.mean(ests))


def full_target_predicted_variance(cells, counts, values, n, missing_mass):
    """Stratified variance over C plus the squared bias of the missing mass."""
    var = c.predicted_variance([values[x] for x in cells], counts, n)
    return var, missing_mass ** 2


def diagnose(name):
    arrays, split, meta, e, offline = r.setup(name)
    out = r.JOBROOT / 'results' / 'diag_v1'
    out.mkdir(parents=True, exist_ok=True)
    n, h, R, J, delta = e.n, e.h, e.radius, e.layers, e.delta
    per_query, cal_rows, ring_rows = [], [], []
    norm_report = {'reference_norm_pcts': np.percentile(e.norms, [1, 5, 25, 50, 75, 95, 99]).tolist(),
                   'query_norm_pcts': np.percentile(np.linalg.norm(arrays['validation'], axis=1),
                                                    [1, 25, 50, 75, 99]).tolist(),
                   'R': R, 'delta': delta, 'R_over_delta': R / delta, 'J': J}
    # Angle-Hamming calibration on reference pairs (offline, train/reference only).
    ref = arrays['reference'].astype(np.float64)
    rng = np.random.default_rng(20260915)
    pi, pj = rng.choice(n, size=2 * CAL_PAIRS, replace=True).reshape(2, -1)
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]
    denom = np.maximum(np.linalg.norm(ref[pi], axis=1) * np.linalg.norm(ref[pj], axis=1), 1e-300)
    cosang = np.clip((ref[pi] * ref[pj]).sum(1) / denom, -1, 1)
    angles = np.degrees(np.arccos(cosang))
    for t in range(e.index.tables_count):
        codes = e.index.point_codes[t]
        dh = np.array([bin(int(codes[a]) ^ int(codes[b])).count('1') for a, b in zip(pi, pj)], dtype=float)
        for lo in range(0, 90, 15):
            m = (angles >= lo) & (angles < lo + 15)
            if m.sum() >= 20:
                cal_rows.append(dict(table=t, angle_bin=lo + 7.5, pairs=int(m.sum()),
                                     mean_hamming=float(dh[m].mean()),
                                     binomial_prediction=e.index.bits * (np.radians(lo + 7.5) / math.pi)))
    for qi, q in enumerate(arrays['validation']):
        d2 = c.sqdist(e.points, q)
        logk = -d2 / (2 * h * h)
        shift = float(logk.max())
        values = np.exp(logk - shift)
        full = float(values.sum() / n)
        inside = d2 <= R ** 2
        codes = e.index.query_codes(q)
        union, _, rings = e.partition(q, codes, R)
        C = np.asarray(sorted(union), dtype=int)
        cmask = np.zeros(n, dtype=bool); cmask[C] = True
        missing = float(values[inside & ~cmask].sum() / n)
        extra = float(values[~inside & cmask].sum() / n)
        tail = float(values[~inside].sum() / n)
        q_norm = float(np.linalg.norm(q))
        bounds = np.linspace(0, R, J + 1)[1:]
        src = c.source
        Tvals = [int(max(e.index.hamming_probe,
                         min(e.index.bits, math.ceil(e.index.bits * src.max_angle_from_radius(q_norm, float(b)) / math.pi))))
                 for b in bounds]
        low0, high0 = src.sphere_layer_bounds(q_norm, float(bounds[0]), delta)
        inner_band = int(((e.norm_layers >= low0) & (e.norm_layers <= high0)).sum())
        for j, ring in enumerate(rings):
            ring_rows.append(dict(query=qi, ring=j, size=len(ring)))
        # fixed-C partitions
        budget_rows = []
        for budget in BUDGETS:
            approx_cells = c.prepare_rings(rings, budget)
            sizes = [len(x) for x in approx_cells]
            counts = c.allocate(sizes, budget)
            boundaries = np.cumsum(sizes)[:-1]
            ordered = C[np.argsort(logk[C], kind='stable')[::-1]] if len(C) else C
            ordered_cells = list(np.split(ordered, boundaries)) if len(C) else []
            # exact-distance strata restricted to C (the best possible use of C)
            labels = np.searchsorted(np.linspace(0, R, J + 1)[1:] ** 2, d2[C], side='left')
            exact_on_C = [C[labels == i] for i in range(J + 1)]
            exact_on_C = [x for x in exact_on_C if len(x)]
            uniform_cells = [C] if len(C) else []
            cases = [('approx', approx_cells, counts),
                     ('ordered', ordered_cells, counts),
                     ('exact_on_C', exact_on_C, None),
                     ('uniform_C', uniform_cells, None)]
            for ps in range(5):
                shuffled = np.random.default_rng(400000 + qi * 100 + ps).permutation(C)
                cases.append(('random', list(np.split(shuffled, boundaries)) if len(C) else [], counts))
            vals_by_case = {}
            for label, cells, alloc in cases:
                if not cells:
                    continue
                szs = [len(x) for x in cells]
                al = alloc if alloc is not None else c.allocate(szs, budget)
                pred = c.predicted_variance([values[x] for x in cells], al, n)
                if label in ('approx', 'uniform_C'):
                    emp, mean_est = empirical_variance(cells, al, values, n, EMP_REPEATS,
                                                        700000 + qi * 100000 + budget * 1000)
                else:
                    emp, mean_est = float('nan'), float('nan')
                vals_by_case[label] = pred
                budget_rows.append(dict(query=qi, budget=budget, grouping=label,
                                        pred_var_scaled=pred, emp_var_scaled=emp,
                                        mean_est_scaled=mean_est))
            if 'uniform_C' in vals_by_case and 'approx' in vals_by_case:
                for row in budget_rows:
                    if row['query'] == qi and row['budget'] == budget and row['grouping'] in vals_by_case:
                        row['ratio_to_uniform_C'] = vals_by_case[row['grouping']] / vals_by_case['uniform_C']
        per_query.append(dict(
            query=qi, q_norm=q_norm, R=R, q_norm_minus_R=q_norm - R,
            candidate_size=len(C), candidate_frac=len(C) / n,
            ball_size=int(inside.sum()), ball_frac=float(inside.mean()),
            id_recall=float(inside[C].sum()) / max(int(inside.sum()), 1),
            mass_recall=float(values[C][inside[C]].sum() / max(values[inside].sum(), 1e-300)),
            full_scaled=full, candidate_scaled=float(values[C].sum() / n),
            missing_scaled=missing, extra_scaled=extra, tail_scaled=tail,
            signed_discrepancy=float(values[C].sum() / n - values[inside].sum() / n),
            T_values=Tvals, distinct_T=len(set(Tvals)),
            innermost_band_size=inner_band, ring_sizes=[len(x) for x in rings],
            budget_rows=budget_rows))
        if (qi + 1) % 20 == 0:
            print(json.dumps({'event': 'diag_progress', 'dataset': name, 'queries': qi + 1}), flush=True)
    br = [row for pq in per_query for row in pq['budget_rows']]
    (out / f'{name}_diag_budgetrows.csv').parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(br).to_csv(out / f'{name}_diag_budgetrows.csv', index=False)
    pd.DataFrame(cal_rows).to_csv(out / f'{name}_angle_hamming.csv', index=False)
    pd.DataFrame(ring_rows).to_csv(out / f'{name}_ring_sizes.csv', index=False)
    slim = [{k: v for k, v in pq.items() if k != 'budget_rows'} for pq in per_query]
    summary = {
        'dataset': name, 'n': n, 'h': h, 'R': R, 'delta': delta, 'J': J,
        'norm_report': norm_report,
        'mean_candidate_frac': float(np.mean([p['candidate_frac'] for p in per_query])),
        'mean_ball_frac': float(np.mean([p['ball_frac'] for p in per_query])),
        'mean_id_recall': float(np.mean([p['id_recall'] for p in per_query])),
        'mean_mass_recall': float(np.mean([p['mass_recall'] for p in per_query])),
        'mean_missing_frac_of_full': float(np.mean([p['missing_scaled'] / p['full_scaled'] for p in per_query])),
        'mean_extra_frac_of_full': float(np.mean([p['extra_scaled'] / p['full_scaled'] for p in per_query])),
        'mean_tail_frac_of_full': float(np.mean([p['tail_scaled'] / p['full_scaled'] for p in per_query])),
        'mean_distinct_T': float(np.mean([p['distinct_T'] for p in per_query])),
        'T_value_histogram': {str(k): int(v) for k, v in zip(
            *np.unique([t for p in per_query for t in p['T_values']], return_counts=True))},
        'mean_innermost_band_frac': float(np.mean([p['innermost_band_size'] for p in per_query])) / n,
        'mean_ratio_approx': float(np.nanmean([row.get('ratio_to_uniform_C') for row in br
                                               if row['grouping'] == 'approx'])),
        'mean_ratio_random': float(np.nanmean([row.get('ratio_to_uniform_C') for row in br
                                               if row['grouping'] == 'random'])),
        'mean_ratio_ordered': float(np.nanmean([row.get('ratio_to_uniform_C') for row in br
                                                if row['grouping'] == 'ordered'])),
        'mean_ratio_exact_on_C': float(np.nanmean([row.get('ratio_to_uniform_C') for row in br
                                                   if row['grouping'] == 'exact_on_C'])),
        'per_query': slim,
        'code_commit': r.git_head(),
    }
    (out / f'{name}_diagnosis.json').write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_query', 'norm_report')}), flush=True)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['isolet', 'cifar10', 'cifar10_gist512', 'amazon']:
        diagnose(name)
