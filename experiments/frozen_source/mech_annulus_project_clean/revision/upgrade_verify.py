"""Verify a completed real run independently of the aggregate analysis."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def verify(root):
    done = json.loads((root / 'DONE.json').read_text())
    assert done['scope'] == 'protocol run', 'Smoke is not evidence'
    report = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / 'raw_queries.jsonl').exists():
            continue
        meta = json.loads((folder / 'OFFLINE.json').read_text())
        complete = json.loads((folder / 'DONE.json').read_text())
        nq = complete['queries']
        budgets = complete['budgets']
        seen = collections.defaultdict(set)
        methods = set()
        work = collections.defaultdict(lambda: [0, 0, 0])
        count = 0
        digest = hashlib.sha256()
        with (folder / 'raw_queries.jsonl').open('rb') as f:
            for line in f:
                digest.update(line)
                r = json.loads(line)
                m = r['method']
                key = (r['budget'], r['seed'], r['repeat'], r['query_id'])
                assert key not in seen[m], 'Duplicated observation'
                seen[m].add(key)
                methods.add(m)
                assert r['raw_row_id'] == meta['query_rows'][r['query_id']]
                assert r['time_ns'] > 0
                assert r['relative_error'] >= 0
                assert r['kernel_evaluations'] >= 0
                if m in ('projected_current', 'projected_fast') or m.startswith('ann_'):
                    assert r['kernel_evaluations'] == r['actual_samples'] == r['budget']
                    assert sum(r['allocation']) + r['H_count'] == r['budget']
                    assert sum(r['layer_sizes']) + r['H_count'] == r['n']
                    assert r.get('unassigned_ns', 0) >= 0
                if m.startswith('ann_'):
                    assert r['ann_duplicate_returned'] >= 0
                    assert r['ann_invalid_returned'] >= 0
                    assert r['actual_H_count'] == r['ann_valid_returned'] - r['ann_duplicate_returned']
                    assert r['ann_valid_returned'] + r['ann_invalid_returned'] == r['requested_neighbors']
                    expected = (r['ann_distance_evaluations'] +
                                r['ann_recomputed_original_distances'] +
                                r['original_remainder_distance_evaluations'] +
                                r['full_dimension_projection_inner_products'])
                    assert expected == r['full_dimension_distance_or_dot_evaluations']
                work[m][0] += r['kernel_evaluations']
                work[m][1] += r['full_dimension_distance_or_dot_evaluations']
                work[m][2] += r['low_dim_score_evaluations']
                count += 1
        expected = {(b, s, rep, q) for b in budgets for s in range(5)
                    for rep in range(3) for q in range(nq)}
        assert all(keys == expected for keys in seen.values()), 'Incomplete method curve'
        checks = [json.loads(line) for line in (folder / 'partition_checks.jsonl').read_text().splitlines()]
        assert len(checks) == nq * len(budgets) * 3
        assert all(x['members_equal'] for x in checks)
        report[folder.name] = dict(queries=nq, budgets=budgets, rows=count,
            methods=sorted(methods), sha256=digest.hexdigest(),
            membership_comparisons=len(checks),
            tie_fallback_comparisons=sum(x['partition_tie_fallback'] for x in checks),
            total_work_by_method=dict(work),
            work_order=['kernel_evaluations', 'original_space_distances_or_dots', 'low_dim_scores'])
    assert report
    return dict(passed=True, scope='Record completeness and accounting; not statistical significance',
                datasets=report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, datasets=list(result['datasets']))))
