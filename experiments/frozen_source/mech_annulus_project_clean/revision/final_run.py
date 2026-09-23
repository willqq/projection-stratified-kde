"""Final frozen test run: 4 datasets x budgets {32..512} x 5 seeds x 3 repeats.

Baselines (exact, uniform_mc, exact_annular) use the original core.Evaluator
code path unchanged. Approx annular uses posting retrieval with the frozen
calibrated configuration. Splits, radius/bandwidth/delta rules and seeds are
identical to round 1. The configuration and model hashes are snapshotted to
assets_final/ BEFORE any test query is evaluated.
"""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import core as c
import run as r
import calib
import fast_core
import search

BUDGETS = [32, 64, 128, 256, 512]
SEEDS = [0, 1, 2, 3, 4]
NQUERIES = 200
REPEATS = 3


def frozen_assets(name, cfg, arrays):
    """Materialize (or verify) the frozen final assets for a dataset."""
    dest = r.JOBROOT / 'assets_final' / name
    marker = dest / 'final_metadata.json'
    key = f"{name}_L{cfg['tables']}_c{cfg['bits']}_e{cfg['epochs']}"
    src = search.SEARCH_ASSETS / key / 'model.pt'
    if marker.exists():
        meta = json.loads(marker.read_text())
        assert meta['config'] == cfg, 'Frozen asset/config mismatch'
        return meta
    dest.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f'Missing searched model {src}; run the search first')
    model = c.source.MultiEncoderContrastiveHash(arrays['reference'].shape[1],
                                                 cfg['tables'], cfg['bits'])
    model.load_state_dict(torch.load(src, map_location='cpu'))
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    index = c.source.MECHHashIndex(model, arrays['reference'], cfg['tables'], cfg['bits'], 2)
    build_seconds = time.perf_counter() - t0
    t_lookup, t_cal_s, cal_meta = None, 0.0, None
    if cfg['t_quantile'] is not None:
        t0 = time.perf_counter()
        t_lookup, mids, T = calib.fit_calibration(index, arrays['train'],
                                                  arrays['reference'], cfg['t_quantile'])
        t_cal_s = time.perf_counter() - t0
        cal_meta = {'bin_mids': mids.tolist(), 'bin_radii': T.tolist(),
                    'quantile': cfg['t_quantile'], 'fit_seed': 20260915}
    meta = {'dataset': name, 'config': cfg, 'model_sha256': r.sha(src),
            'model_source': str(src), 'build_seconds': build_seconds,
            'calibration_seconds': t_cal_s, 'calibration': cal_meta,
            'code_commit': r.git_head(), 'snapshot_time': time.time()}
    shutil.copy2(src, dest / 'model.pt')
    marker.write_text(json.dumps(meta, indent=2))
    return meta


def build(name, cfg):
    arrays, split, meta, _, _ = r.setup(name)
    assets = frozen_assets(name, cfg, arrays)
    model = c.source.MultiEncoderContrastiveHash(arrays['reference'].shape[1],
                                                 cfg['tables'], cfg['bits'])
    model.load_state_dict(torch.load(r.JOBROOT / 'assets_final' / name / 'model.pt',
                                     map_location='cpu'))
    torch.set_num_threads(8)
    index = c.source.MECHHashIndex(model, arrays['reference'], cfg['tables'], cfg['bits'], 2)
    t_lookup = None
    if cfg['t_quantile'] is not None:
        t_lookup, _, _ = calib.fit_calibration(index, arrays['train'],
                                               arrays['reference'], cfg['t_quantile'])
    ev = fast_core.FastEvaluator(arrays['reference'], meta['bandwidth'], meta['radius'],
                                 cfg['layers'], index, meta['delta'] * cfg['delta_scale'],
                                 cfg['min_collisions'], t_lookup=t_lookup)
    ev_base = c.Evaluator(arrays['reference'], meta['bandwidth'], meta['radius'], 8,
                          None, None, 2)
    return arrays, split, meta, ev, ev_base, assets


def benchmark(name, cfg, outdir):
    arrays, split, meta, ev, ev_base, assets = build(name, cfg)
    queries = arrays['test'][:NQUERIES]
    for method in ['exact', 'uniform_mc', 'exact_annular', 'approx_annular_mech']:
        target = ev if method == 'approx_annular_mech' else ev_base
        target.evaluate(method, queries[0], BUDGETS[0], np.random.default_rng(987))
    rows, timings = [], []
    for qi, q in enumerate(queries):
        t = time.perf_counter_ns()
        truth = c.exact_log_mean(ev.points, q, ev.h)
        truth_time = time.perf_counter_ns() - t
        order = ([m for m in ['exact', 'uniform_mc', 'exact_annular', 'approx_annular_mech']
                  [qi % 4:]] + ['exact', 'uniform_mc', 'exact_annular', 'approx_annular_mech'])[:4]
        for method in order:
            target = ev if method == 'approx_annular_mech' else ev_base
            for budget in ([0] if method == 'exact' else BUDGETS):
                for seed in ([0] if method == 'exact' else SEEDS):
                    results = []
                    for rep in range(REPEATS):
                        result = target.evaluate(method, q, max(1, budget),
                                                 np.random.default_rng(seed + qi * 100003))
                        results.append(result)
                        timing = {k: v for k, v in result.items() if k.endswith('_ns')}
                        timings.append(dict(dataset=name, query_id=qi,
                                            raw_row_id=int(split['test'][qi]),
                                            method=method, budget=budget, seed=seed,
                                            repeat=rep, **timing))
                    result = results[0].copy()
                    result['time_ns'] = float(np.median([x_['time_ns'] for x_ in results]))
                    for key in [k for k in results[0] if k.endswith('_ns')]:
                        result[key] = float(np.median([x_[key] for x_ in results]))
                    result.update(c.errors(result.pop('log_estimate'), truth))
                    result['layer_sizes'] = json.dumps(result['layer_sizes'])
                    result['allocation'] = json.dumps(result['allocation'])
                    rows.append(dict(dataset=name, phase='formal', query_id=qi,
                                     raw_row_id=int(split['test'][qi]), method=method,
                                     budget=budget, seed=seed, truth_time_ns=truth_time,
                                     **result))
        if (qi + 1) % 20 == 0:
            print(json.dumps({'event': 'progress', 'dataset': name, 'queries': qi + 1}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / f'{name}_queries.csv', index=False)
    pd.DataFrame(timings).to_csv(outdir / f'{name}_timings.csv', index=False)
    c.aggregate(frame).to_csv(outdir / f'{name}_summary.csv', index=False)
    (outdir / f'{name}_DONE.json').write_text(json.dumps(
        {'rows': len(rows), 'code_commit': r.git_head(), 'config': cfg,
         'assets': {k: assets[k] for k in ('model_sha256', 'code_commit')},
         'completed_at': time.time()}))
    print(json.dumps({'event': 'complete', 'dataset': name, 'rows': len(rows)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--run-id', default='basic_final')
    ap.add_argument('--config', required=True, help='frozen final_config.json')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    outdir = r.JOBROOT / 'results' / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / f'{args.dataset}_queries.csv').exists():
        raise RuntimeError('Output exists; refusing to overwrite')
    command = {'argv': ['final_run.py', args.dataset], 'code_commit': r.git_head(),
               'config': cfg, 'budgets': BUDGETS, 'sampling_seeds': SEEDS,
               'test_queries': NQUERIES, 'timing_repeats': REPEATS, 'threads': 8}
    (outdir / f'{args.dataset}_command.json').write_text(json.dumps(command, indent=2))
    benchmark(args.dataset, cfg, outdir)


if __name__ == '__main__':
    main()
