"""Exact finite-population checks plus budget and implementation edge cases."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
from sprint_estimator import estimate_strata, allocate_counts, _expand_complement_positions


class ScriptedRNG:
    def __init__(self, choices):
        self.choices = [np.asarray(x, dtype=np.int64) for x in choices]
        self.used = 0
    def choice(self, a, size, replace):
        assert not replace
        if self.used < len(self.choices):
            ids = self.choices[self.used]
        else:
            ids = np.arange(size, dtype=np.int64)
        self.used += 1
        assert len(ids) == size and len(np.unique(ids)) == size
        assert np.all((ids >= 0) & (ids < a))
        return ids


def kernel(values):
    values = np.asarray(values, dtype=float)
    logs = np.full(len(values), -np.inf)
    logs[values > 0] = np.log(values[values > 0])
    def evaluate(ids):
        return logs[ids]
    return evaluate


def invariants(result, n, budget):
    ids = result['selected_ids']
    assert len(ids) == min(n, budget) == result['actual_samples']
    assert len(np.unique(ids)) == len(ids)
    assert sum(result['kernel_call_sizes']) == len(ids)
    assert result['kernel_calls'] <= 2
    assert result['H_count'] + int(result['cell_sizes'].sum()) == n
    assert result['H_count'] + int((result['pilot_counts'] + result['main_counts']).sum()) == len(ids)
    remaining = result['cell_sizes'] - result['pilot_counts']
    assert np.all(result['main_counts'][remaining > 0] >= 1)
    assert np.all(result['main_counts'] <= remaining)
    assert result['unassigned_ns'] >= 0
    for pilots, main in zip(result['pilot_ids'], result['main_ids']):
        assert not np.intersect1d(pilots, main).size


def conditional_enumeration(values, label, rows):
    n = 11
    H = np.array([0])
    cells = [np.arange(1, 6), np.arange(6, 11)]
    pilot_options = list(itertools.combinations(range(5), 2))
    target = float(np.mean(values))
    kernel_fn = kernel(values)
    seen_allocations = set()
    total_outcomes = 0
    for pa, pb in itertools.product(pilot_options, repeat=2):
        probe = estimate_strata(n, H, cells, 8, ScriptedRNG([pa, pb]), kernel_fn,
                                 'neyman', validate=True)
        counts = probe['main_counts']
        assert probe['effective_allocation'] == 'neyman'
        seen_allocations.add(tuple(map(int, counts)))
        options = [list(itertools.combinations(range(3), int(b))) for b in counts]
        estimates = []
        for ma, mb in itertools.product(*options):
            result = estimate_strata(n, H, cells, 8, ScriptedRNG([pa, pb, ma, mb]),
                                     kernel_fn, 'neyman', validate=True)
            invariants(result, n, 8)
            estimates.append(math.exp(result['log_estimate']))
        mean = float(np.mean(estimates))
        error = abs(mean - target)
        assert error <= 1e-13 * max(1., abs(target)), (label, pa, pb, mean, target)
        rows.append({'case': label, 'pilot_cell_0': str(pa), 'pilot_cell_1': str(pb),
                     'main_counts': str(tuple(map(int, counts))), 'main_outcomes': len(estimates),
                     'conditional_mean': mean, 'full_target': target, 'absolute_difference': error,
                     'pass': True})
        total_outcomes += len(estimates)
    return {'case': label, 'pilot_conditions': 100, 'enumerated_main_outcomes': total_outcomes,
            'distinct_adaptive_main_allocations': sorted(map(list, seen_allocations)), 'pass': True}


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    conditional_rows = []
    checks.append(conditional_enumeration(np.array([.7,.05,.1,.2,.8,1.,.01,.02,.03,.04,.9]),
                                          'heterogeneous', conditional_rows))
    checks.append(conditional_enumeration(np.array([0,0,0,0,0,1,0,0,0,0,0], dtype=float),
                                          'single_spike', conditional_rows))
    checks.append(conditional_enumeration(np.full(11, .125), 'constant', conditional_rows))
    # Exact proportional expectation under all within-layer sample combinations.
    vals = np.array([.5,.01,.05,.1,.4,.6,.8,1.])
    estimates = []
    for a in itertools.combinations(range(3), 1):
        for b in itertools.combinations(range(4), 2):
            r = estimate_strata(8, [0], [np.arange(1,4),np.arange(4,8)], 4,
                                ScriptedRNG([a,b]), kernel(vals), validate=True)
            invariants(r, 8, 4)
            estimates.append(math.exp(r['log_estimate']))
    assert abs(np.mean(estimates) - vals.mean()) < 1e-14
    checks.append({'case':'proportional_exact_expectation', 'outcomes':len(estimates),
                   'absolute_difference':abs(float(np.mean(estimates)-vals.mean())), 'pass':True})
    # Compressed complement mapping has no omission, duplicate, or full-size mask.
    maps = 0
    for n in range(1,15):
        for k in range(min(2,n)+1):
            for excluded in itertools.combinations(range(n), k):
                got = _expand_complement_positions(np.arange(n-k), np.asarray(excluded,dtype=int))
                expected = np.asarray([i for i in range(n) if i not in excluded])
                assert np.array_equal(got, expected)
                maps += 1
    checks.append({'case':'complement_position_bijection', 'cases':maps, 'pass':True})
    # All these labels indicate how callers form cells; estimator sees a full partition.
    cases = [
        ('empty_C_one_residual',20,[],[np.arange(20)],7),
        ('C_equals_P',20,[],list(np.array_split(np.arange(20),4)),9),
        ('singleton_H',1,[0],[],1),
        ('singleton_cell',1,[],[np.array([0])],1),
        ('empty_cells_ignored',4,[],[np.array([],dtype=int),np.arange(4)],2),
        ('budget_exceeds_n',7,[0,1],[np.arange(2,5),np.arange(5,7)],32),
        ('budget_equals_n',7,[0,1],[np.arange(2,5),np.arange(5,7)],7),
        ('M32_neyman',100,np.arange(8),list(np.array_split(np.arange(8,100),5)),32),
        ('M32_pilot_fallback',40,np.arange(8),list(np.array_split(np.arange(8,40),16)),32),
        ('merge_before_pilot',40,np.arange(8),list(np.array_split(np.arange(8,40),16)),10),
        ('minimal_one_sample',20,[],list(np.array_split(np.arange(20),4)),1),
        ('singletons_mixed',20,[0],[np.array([1]),np.arange(2,6),np.arange(6,20)],8),
    ]
    for label,n,H,cells,budget in cases:
        for mode in ['proportional','neyman']:
            for seed in range(10):
                trace = []
                def constant_kernel(ids):
                    trace.extend(map(int,ids))
                    return np.full(len(ids), -1000.)
                r = estimate_strata(n,H,cells,budget,np.random.default_rng(seed),constant_kernel,
                                    mode,validate=True)
                invariants(r,n,budget)
                assert len(trace) == len(set(trace)) == min(n,budget)
                assert abs(r['log_estimate'] + 1000.) < 1e-11
                if label == 'M32_pilot_fallback':
                    assert mode == 'proportional' or r['effective_allocation'] == 'proportional'
                if label == 'merge_before_pilot':
                    assert r['merged_cells']
            checks.append({'case':label, 'allocation':mode, 'seeds':10, 'pass':True})
    # Arbitrary allocations always consume budget and obey capacities/minima.
    alloc_cases = 0
    for n in range(1,25):
        for groups in range(1,min(7,n)+1):
            sizes = np.asarray([len(x) for x in np.array_split(np.arange(n),groups)])
            for budget in range(groups,n+3):
                for weights in [None, np.zeros(groups), np.arange(groups,dtype=float), np.ones(groups)]:
                    a = allocate_counts(sizes,budget,weights)
                    assert a.sum() == min(n,budget) and np.all(a>=1) and np.all(a<=sizes)
                    alloc_cases += 1
    checks.append({'case':'allocation_capacity_minimum_budget', 'cases':alloc_cases,'pass':True})
    # Invalid coverage must fail in explicit validation mode.
    bad = [(4,[0],[[1,1,3]],2), (4,[0],[[1,2,4]],2), (4,[0,1],[[2,3]],1)]
    for n,H,cells,budget in bad:
        try:
            estimate_strata(n,H,cells,budget,np.random.default_rng(0),kernel(np.ones(n)),validate=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid partition or budget accepted')
    checks.append({'case':'invalid_partition_or_H_budget_rejected','cases':len(bad),'pass':True})
    # Equal-valued pilot logs may all be -inf; positive nonpilot mass remains sampleable.
    def all_zero(ids):
        return np.full(len(ids), -np.inf)
    r = estimate_strata(30,[],list(np.array_split(np.arange(30),4)),20,
                        np.random.default_rng(5),all_zero,'neyman',validate=True)
    invariants(r,30,20)
    assert r['log_estimate'] == -np.inf
    checks.append({'case':'all_zero_kernel_no_nan','pass':True})
    with (out/'conditional_expectation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(conditional_rows[0]));writer.writeheader();writer.writerows(conditional_rows)
    source = Path(__file__).with_name('sprint_estimator.py')
    report={'pass':True,'python':sys.version,'numpy':np.__version__,
            'estimator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checks':checks,'conditional_cases':len(conditional_rows),
            'max_conditional_absolute_error':max(r['absolute_difference'] for r in conditional_rows),
            'note':'Every pilot condition enumerates all possible formal uniform subsets; no Monte Carlo approximation to the conditional expectation.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
