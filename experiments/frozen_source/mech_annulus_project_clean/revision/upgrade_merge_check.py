"""Apply the preregistered merge criteria to completed formal summaries."""
import argparse
import json
from pathlib import Path
import pandas as pd
from upgrade_analyze import adjacent


def run(folder):
    def read(name, columns):
        try:
            return pd.read_csv(folder / name)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=columns)
    A = read('path_a_timing_bootstrap.csv', ['dataset', 'phase', 'n', 'budget', 'relative_reduction'])
    B = read('ann_H_paired_bootstrap.csv', [])
    V = read('ann_H_variance.csv', [])
    output = dict(path_a=False, path_b=False, qualified_a=[],
                  statistical_b=[], computational_b=[], qualified_b=[])
    for keys, z in A[A.phase != 'validation'].groupby(['dataset', 'phase', 'n']):
        good = z[z.relative_reduction >= .20]
        if adjacent(good.budget):
            output['path_a'] = True
            output['qualified_a'].append(dict(dataset=keys[0], phase=keys[1],
                n=int(keys[2]), budgets=good.budget.tolist()))
    statkeys = set()
    if len(B):
        joined = B.merge(V, on=['dataset', 'phase', 'n', 'anchor', 'budget'],
                         suffixes=('', '_variance'), validate='one_to_one')
        for keys, z in joined[joined.phase != 'validation'].groupby(['dataset', 'phase', 'n', 'anchor']):
            good = z[(z.relative_reduction >= .10) & (z.ci_low > 0) & (z.median_variance_ratio <= .90)]
            if adjacent(good.budget):
                statkeys.add(keys)
                output['statistical_b'].append(dict(dataset=keys[0], phase=keys[1],
                    n=int(keys[2]), anchor=keys[3], budgets=good.budget.tolist()))
    # The strongest measured baseline envelope is available at each target.
    # Include the faster equivalent implementation too when measured; this
    # prevents attributing Path A's implementation gain to Path B's estimator.
    rows = []
    for filename, axis, value, order in [
        ('fixed_error_latency.csv', 'error_target', 'latency_ms', [.05, .10, .15, .20]),
        ('fixed_time_error.csv', 'time_cap_ms', 'error', [.125, .25, .5, 1., 2., 4.])]:
        df = pd.read_csv(folder / filename)
        for keys, z in df[df.phase != 'validation'].groupby(['dataset', 'phase', 'n']):
            is_control = z.method.str.startswith('ann_')
            base = z[~is_control & z.attained]
            methods = z[z.method.str.startswith('ann_') & z.method.str.endswith('_multi')].method.unique()
            for method in methods:
                wins = set()
                for target in order:
                    eligible = base[base[axis] == target]
                    mine = z[(z.method == method) & (z[axis] == target) & z.attained]
                    baseline = eligible.sort_values(value).iloc[0] if len(eligible) else None
                    candidate = mine.iloc[0] if len(mine) else None
                    gain = (1 - candidate[value] / baseline[value]
                            if baseline is not None and candidate is not None and baseline[value] > 0 else None)
                    if gain is not None and gain >= .10:
                        wins.add(target)
                    rows.append(dict(dataset=keys[0], phase=keys[1], n=int(keys[2]),
                        method=method, axis=axis, target=target,
                        baseline_method=baseline.method if baseline is not None else None,
                        baseline_budget=baseline.budget if baseline is not None else None,
                        baseline_value=baseline[value] if baseline is not None else None,
                        candidate_budget=candidate.budget if candidate is not None else None,
                        candidate_value=candidate[value] if candidate is not None else None,
                        relative_gain=gain))
                if any(x in wins and y in wins for x,y in zip(order[:-1],order[1:])):
                    anchor = method.split('_')[1]
                    record = dict(dataset=keys[0], phase=keys[1], n=int(keys[2]),
                                  anchor=anchor, axis=axis, targets=sorted(wins))
                    output['computational_b'].append(record)
                    if (*keys, anchor) in statkeys:
                        output['path_b'] = True
                        output['qualified_b'].append(record)
    output['decision'] = ('C' if output['path_a'] and output['path_b'] else
                          'A' if output['path_a'] else 'B' if output['path_b'] else 'D')
    output['interpretation'] = 'Project decision criteria, not conference acceptance thresholds'
    pd.DataFrame(rows).to_csv(folder / 'strongest_envelope_comparison.csv', index=False)
    (folder / 'MERGE_GATE.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    run(p.parse_args().folder)
