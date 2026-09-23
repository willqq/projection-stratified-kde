"""Check Path A membership on existing validation score caches only.

This is not a new dataset run and supplies no complete-query performance claim.
"""
import csv
import json
from pathlib import Path
import numpy as np
from upgrade_partition import split_scores, profile_original
from sprint_estimator import allocate_counts


def main(root,out):
    rng=np.random.default_rng(20260920)
    rows=[]
    for dataset in ['isolet','cifar10','cifar10_gist512','amazon']:
        with np.load(root/(dataset+'_cache.npz')) as data:
            for query in range(80):
                score=data['q%d_projected'%query]
                values=data['q%d_values'%query]
                ids=np.arange(len(score),dtype=np.int64)
                priority=rng.random(len(ids))
                for budget in [32,64,128,256,512]:
                    k=budget//4
                    H,a,_=profile_original(score,ids,k,8)
                    H2,b,meta=split_scores(score,ids,k,8)
                    assert np.array_equal(np.sort(H),np.sort(H2))
                    assert len(a)==len(b)
                    assert all(np.array_equal(np.sort(x),np.sort(y)) for x,y in zip(a,b))
                    counts=allocate_counts(list(map(len,a)),budget-k)
                    assert np.array_equal(counts,allocate_counts(list(map(len,b)),budget-k))
                    def coupled(cells):
                        selected=[np.sort(c[np.argsort(priority[c])[:m]]) for c,m in zip(cells,counts)]
                        result=values[np.sort(H)].sum()
                        result+=sum(len(c)/m*values[s].sum() for c,m,s in zip(cells,counts,selected))
                        return result/len(ids),selected
                    x,xids=coupled(a);y,yids=coupled(b)
                    assert x==y
                    assert all(np.array_equal(x,y) for x,y in zip(xids,yids))
                    rows.append(dict(dataset=dataset,query=query,
                        raw_row_id=int(data['q%d_raw_row_id'%query]),n=len(ids),budget=budget,
                        H_count=k,all_cell_members_equal=True,allocation_equal=True,
                        coupled_estimate_equal=True,tie_fallback=meta['partition_tie_fallback']))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(validation_queries=320,budget_cases=len(rows),
        all_passed=True,tie_fallbacks=sum(x['tie_fallback'] for x in rows),
        scope='Frozen validation score cache; no new full-query timing'),indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.root,args.out)
