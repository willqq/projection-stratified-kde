"""Membership, ID-coupling and exact finite-population distribution checks."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from upgrade_partition import split_scores, profile_original
from sprint_estimator import allocate_counts, estimate_strata


def canonical(H, cells):
    return tuple(sorted(H)), tuple(tuple(sorted(c)) for c in cells)


def check_case(scores, ids, k, J, budget, rng):
    oldH, old, _ = profile_original(scores, ids, k, J)
    newH, new, detail = split_scores(scores, ids, k, J)
    assert canonical(oldH, old) == canonical(newH, new)
    assert len(np.unique(np.concatenate([newH]+new))) == len(ids)
    if new and budget-k >= len(new):
        alloc = allocate_counts(list(map(len, old)), budget-k)
        assert np.array_equal(alloc, allocate_counts(list(map(len, new)), budget-k))
        priority = rng.random(int(max(ids))+1)
        # IID continuous priorities indexed by population ID couple exactly the
        # same uniform fixed-size samples, independent of internal cell order.
        a = [np.sort(c[np.argsort(priority[c])[:m]]) for c,m in zip(old,alloc)]
        b = [np.sort(c[np.argsort(priority[c])[:m]]) for c,m in zip(new,alloc)]
        assert all(np.array_equal(x,y) for x,y in zip(a,b))
    return detail['partition_tie_fallback']


def run():
    rng=np.random.default_rng(20260920)
    cases=fallbacks=0
    for n in [1,2,7,9,31,32,63,65,127,128,720,1800,20000]:
        ids=rng.permutation(n)
        for mode in ['continuous','quantized','equal','monotone']:
            scores=(rng.normal(size=n) if mode=='continuous' else
                    rng.integers(0,7,size=n).astype(float) if mode=='quantized' else
                    np.ones(n) if mode=='equal' else np.arange(n,dtype=float))
            for budget in [32,64,128,256,512]:
                k=min(budget//4,n,max(0,budget-1))
                for J in [1,4,8]:
                    fallbacks+=check_case(scores,ids,k,J,budget,rng);cases+=1
    # Every possible equal-size sample combination under both layouts yields
    # the same estimate multiset, hence exactly the same distribution.
    n=13;k=1;J=3;budget=7
    scores=rng.normal(size=n);ids=rng.permutation(n)
    H,a,_=profile_original(scores,ids,k,J)
    H2,b,_=split_scores(scores,ids,k,J)
    values=np.exp(-np.linspace(0,5,n))
    counts=allocate_counts(list(map(len,a)),budget-k)
    def support(cells):
        result=[]
        choices=[list(itertools.combinations(c,int(m))) for c,m in zip(cells,counts)]
        for selection in itertools.product(*choices):
            v=(sum(values[H])+sum(len(c)/m*sum(values[list(s)])
                for c,m,s in zip(cells,counts,selection)))/n
            result.append(v)
        return np.sort(result)
    x=support(a);y=support(b)
    assert np.allclose(x,y,rtol=0,atol=1e-15)
    expected=values.mean()
    variance=sum(len(c)**2/m*(1-m/len(c))*np.var(values[c],ddof=1)
                 for c,m in zip(a,counts))/n**2
    assert abs(x.mean()-expected)<1e-14
    assert abs(x.var()-variance)<1e-14
    # Exercise the unchanged production sampling/weighting code as well.
    draws=[]
    for seed in range(4096):
        row=estimate_strata(n,H2,b,budget,np.random.default_rng(seed),
                            lambda selected:np.log(values[selected]),validate=True)
        assert row['actual_samples']==budget
        assert len(np.unique(row['selected_ids']))==budget
        draws.append(np.exp(row['log_estimate']))
    se=math.sqrt(variance/len(draws))
    assert abs(np.mean(draws)-expected)<5*se
    report=dict(membership_cases=cases,tie_fallback_cases=fallbacks,
        exact_distribution_support_size=len(x),true_mean=expected,
        enumerated_mean=float(x.mean()),analytical_variance=variance,
        enumerated_variance=float(x.var()),production_repetitions=len(draws),
        production_mean=float(np.mean(draws)),production_variance=float(np.var(draws)),
        scope='Synthetic correctness only; no real-data error or timing claim')
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
