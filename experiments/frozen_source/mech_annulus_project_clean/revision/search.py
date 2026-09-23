"""Validation-only staged configuration search for the approximate annuli.

Stage A: c in {5,8,12,16} x T in {default, calibrated} (defaults L=12,b=2,delta,J=8).
Stage B: one-dimension-at-a-time around the stage-A winner (L, b, delta, J, beta).
Selection: mean rank of validation mean relative error over budgets 64/128/256
across the four datasets (shared configuration only); latency tie-break.
No test query is read anywhere in this module.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import core as c
import run as r
import calib
import fast_core

SEARCH_ASSETS = r.JOBROOT / 'search_assets'
BUDGETS = [64, 128, 256]
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ['isolet', 'cifar10', 'cifar10_gist512', 'amazon']


def setup_search(name, tables, bits, epochs=None):
    """Like run.setup but parameterized; caches models under search_assets/."""
    arrays, split, meta, _, _ = r.setup(name)  # loads frozen meta + splits
    epochs = epochs or r.TRAIN['epochs']
    key = f'{name}_L{tables}_c{bits}_e{epochs}'
    folder = SEARCH_ASSETS / key
    if not (folder / 'model.pt').exists():
        folder.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(r.MODEL_SEED)
        np.random.seed(r.MODEL_SEED)
        torch.use_deterministic_algorithms(True)
        model = c.source.train_mech(arrays['train'], tables, bits, epochs,
                                     r.TRAIN['batch_size'], r.TRAIN['lr'],
                                     r.TRAIN['alpha'], r.TRAIN['beta'],
                                     r.TRAIN['gamma_balance'],
                                     r.TRAIN['gamma_decorrelation'], 'cpu')
        torch.save(model.state_dict(), folder / 'model.pt')
    model = c.source.MultiEncoderContrastiveHash(arrays['reference'].shape[1], tables, bits)
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu'))
    return arrays, split, meta, model


def build_evaluator(name, tables, bits, epochs=None, delta_scale=1.0, layers=8,
                    min_collisions=2, t_quantile=None):
    arrays, split, meta, model = setup_search(name, tables, bits, epochs)
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    index = c.source.MECHHashIndex(model, arrays['reference'], tables, bits, 2)
    build_s = time.perf_counter() - t0
    t_lookup = None
    cal_s = 0.0
    if t_quantile is not None:
        t0 = time.perf_counter()
        t_lookup, mids, T = calib.fit_calibration(index, arrays['train'],
                                                  arrays['reference'], t_quantile)
        cal_s = time.perf_counter() - t0
    ev = fast_core.FastEvaluator(arrays['reference'], meta['bandwidth'], meta['radius'],
                                 layers, index, meta['delta'] * delta_scale, min_collisions,
                                 t_lookup=t_lookup)
    ev.dataset_name = name
    return arrays, ev, dict(build_seconds=build_s, calibration_seconds=cal_s,
                            t_bins=(mids.tolist(), T.tolist()) if t_lookup is not None else None)


def evaluate_config(name, tables, bits, epochs=None, delta_scale=1.0, layers=8,
                    min_collisions=2, t_quantile=None, include_baselines=False,
                    fixed_c_diagnostics=True):
    arrays, ev, info = build_evaluator(name, tables, bits, epochs, delta_scale,
                                       layers, min_collisions, t_quantile)
    queries = arrays['validation']
    truths = [c.exact_log_mean(ev.points, q, ev.h) for q in queries]
    rows = []
    methods = ['approx_annular_mech'] + (['uniform_mc', 'exact_annular'] if include_baselines else [])
    for method in methods:
        ev0 = ev if method == 'approx_annular_mech' else c.Evaluator(
            ev.points, ev.h, ev.radius, 8,
            ev.index if method == 'approx_annular_mech' else None,
            ev.delta if method == 'approx_annular_mech' else None, 2)
        for qi, q in enumerate(queries):
            for budget in BUDGETS:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed + qi * 100003)
                    res = ev0.evaluate(method, q, budget, rng)
                    err = c.errors(res.pop('log_estimate'), truths[qi])
                    rows.append(dict(dataset=name, method=method, query_id=qi,
                                     budget=budget, seed=seed, **err,
                                     **{k: v for k, v in res.items()
                                        if not isinstance(v, (list, np.ndarray))}))
    frame = pd.DataFrame(rows)
    diag = fixed_c_summary(arrays, ev) if fixed_c_diagnostics else {}
    return frame, info, diag


def fixed_c_summary(arrays, ev, budgets=(128,), repeats=60):
    """Predicted conditional variance ratio approx/uniform on C, validation."""
    ratios, recalls, cands = [], [], []
    for qi, q in enumerate(arrays['validation'][:20]):
        d2 = c.sqdist(ev.points, q)
        logk = -d2 / (2 * ev.h * ev.h)
        shift = float(logk.max())
        values = np.exp(logk - shift)
        codes = ev.index.query_codes(np.asarray(q, dtype=np.float32))
        union, _, rings = ev.partition(q, codes, ev.radius)
        C = np.asarray(sorted(union), dtype=int)
        if not len(C):
            continue
        inside = d2 <= ev.radius ** 2
        recalls.append(float(values[C][inside[C]].sum() / max(values[inside].sum(), 1e-300)))
        cands.append(len(C) / ev.n)
        for budget in budgets:
            approx = c.prepare_rings(rings, budget)
            pred_approx = c.predicted_variance([values[x] for x in approx],
                                               c.allocate([len(x) for x in approx], budget), ev.n)
            pred_unif = c.predicted_variance([values[C]],
                                             c.allocate([len(C)], budget), ev.n)
            if pred_unif > 0:
                ratios.append(pred_approx / pred_unif)
    return dict(variance_ratio_mean=float(np.mean(ratios)) if ratios else float('nan'),
                variance_ratio_median=float(np.median(ratios)) if ratios else float('nan'),
                mass_recall_mean=float(np.mean(recalls)) if recalls else float('nan'),
                candidate_frac_mean=float(np.mean(cands)) if cands else float('nan'))


def summarize(frame):
    s = frame.groupby(['dataset', 'method', 'budget'], as_index=False).agg(
        mare=('relative_error', 'mean'),
        median_ms=('time_ns', lambda x: x.median() / 1e6),
        mean_ms=('time_ns', lambda x: x.mean() / 1e6))
    return s


def config_key(tables, bits, epochs, delta_scale, layers, min_collisions, t_quantile):
    return (f'L{tables}_c{bits}_e{epochs or r.TRAIN["epochs"]}_d{delta_scale:g}'
            f'_J{layers}_b{min_collisions}_'
            + (f'cal{t_quantile:g}' if t_quantile is not None else 'Tdefault'))


def run_stage(stage, only_dataset=None, extra=None):
    outdir = r.JOBROOT / 'results' / 'search_v1'
    outdir.mkdir(parents=True, exist_ok=True)
    configs = []
    if stage == 'A':
        for bits in (5, 8, 12, 16):
            for tq in (None, 0.8):
                configs.append(dict(tables=12, bits=bits, delta_scale=1.0, layers=8,
                                    min_collisions=2, t_quantile=tq))
    elif stage == 'B' and extra:
        base = json.loads(extra)
        for dim, vals in base['vary'].items():
            for v in vals:
                cfg = dict(base['base'])
                cfg[dim] = v
                configs.append(cfg)
    all_rows = []
    for cfg in configs:
        key = config_key(cfg['tables'], cfg['bits'], cfg.get('epochs'), cfg['delta_scale'],
                         cfg['layers'], cfg['min_collisions'], cfg['t_quantile'])
        for name in ([only_dataset] if only_dataset else DATASETS):
            t0 = time.perf_counter()
            frame, info, diag = evaluate_config(
                name, cfg['tables'], cfg['bits'], cfg.get('epochs'), cfg['delta_scale'],
                cfg['layers'], cfg['min_collisions'], cfg['t_quantile'])
            summ = summarize(frame)
            rec = dict(config=key, dataset=name, seconds=time.perf_counter() - t0,
                       mare_by_budget={int(b): float(s.mare.iloc[0]) for b, s in
                                      summ[summ.method == 'approx_annular_mech'].groupby('budget')},
                       median_ms_by_budget={int(b): float(s.median_ms.iloc[0]) for b, s in
                                            summ[summ.method == 'approx_annular_mech'].groupby('budget')},
                       **{f'fc_{k}': v for k, v in diag.items()},
                       build_seconds=info['build_seconds'],
                       calibration_seconds=info['calibration_seconds'],
                       t_bins=info['t_bins'], **cfg)
            all_rows.append(rec)
            print(json.dumps({'event': 'config_done', 'config': key, 'dataset': name,
                              'mare': rec['mare_by_budget'],
                              'variance_ratio': rec.get('fc_variance_ratio_median'),
                              'mass_recall': rec.get('fc_mass_recall_mean'),
                              'cand_frac': rec.get('fc_candidate_frac_mean')}), flush=True)
            frame.to_csv(outdir / f'{key}__{name}_rows.csv', index=False)
    pd.DataFrame(all_rows).to_csv(outdir / f'stage_{stage}_{"extra" if stage == "B" else ""}summary.csv', index=False)
    # Shared-config ranking across datasets.
    df = pd.DataFrame(all_rows)
    df['mare_mean'] = df['mare_by_budget'].apply(lambda d: float(np.mean(list(d.values()))))
    per = df.groupby(['config', 'dataset'], as_index=False)['mare_mean'].mean()
    per['rank'] = per.groupby('dataset')['mare_mean'].rank(method='average')
    ranks = per.groupby('config', as_index=False)['rank'].mean().set_index('config')['rank']
    print(json.dumps({'event': 'stage_ranking', 'stage': stage,
                      'ranking': ranks.sort_values().round(6).to_dict()}), flush=True)
    ranks.reset_index().rename(columns={'rank': 'mean_rank'}).to_csv(
        outdir / f'stage_{stage}_ranking.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['A', 'B'], required=True)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--extra', default=None, help='JSON with base config and vary dims')
    args = ap.parse_args()
    run_stage(args.stage, args.dataset, args.extra)
