import itertools
import math
import numpy as np
import pandas as pd
import pytest
import core as c


def test_normalization_and_full_census():
    p = np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
    q = np.zeros(2)
    e = c.Evaluator(p, 1., 1., layers=2)
    truth = np.exp(-np.arange(4)**2/2).mean()
    for method in ['exact','uniform_mc','exact_annular']:
        result = e.evaluate(method,q,100,np.random.default_rng(0))
        assert np.exp(result['log_estimate']) == pytest.approx(truth)
        assert result['actual_samples'] == 4
    # Candidate normalization still uses full n, not |C|.
    assert np.exp(c.log_weighted_estimate(np.zeros(2),np.zeros(2),4)) == .5


def test_budget_caps_merging_and_empty():
    for sizes in [[1,100],[1,1,1],[2,5,30],[200]]:
        rings=[]
        start=0
        for size in sizes:
            rings.append(np.arange(start,start+size)); start+=size
        for budget in [1,2,3,8,128,512]:
            cells=c.prepare_rings(rings,budget)
            counts=c.allocate([len(x) for x in cells],budget)
            assert counts.sum()==min(budget,sum(sizes))
            assert np.all(counts>=1)
            assert np.all(counts<=np.array([len(x) for x in cells]))
            ids,_=c.sample_ids(cells,counts,np.random.default_rng(3))
            assert len(ids)==len(set(ids))
    assert c.log_weighted_estimate(np.array([]),np.array([]),3)==-math.inf


def test_conditional_mean_and_finite_population_variance_exact_enumeration():
    values=[np.array([.1,.4,.9]),np.array([.2,.8])]
    counts=[2,1]
    n=9
    outcomes=[]
    for a in itertools.combinations(values[0],2):
        for b in itertools.combinations(values[1],1):
            outcomes.append(np.exp(c.log_weighted_estimate(np.log(np.r_[a,b]), np.log([1.5,1.5,2.]), n)))
    assert np.mean(outcomes)==pytest.approx(sum(x.sum() for x in values)/n)
    assert np.var(outcomes)==pytest.approx(c.predicted_variance(values,counts,n))
    assert c.predicted_variance(values,[3,2],n)==0


def test_geometry_boundaries():
    for a,r,expected in [(1,0,0),(1,1,math.pi/2),(1,2,math.pi),(0,0,math.pi),(0,2,math.pi)]:
        assert c.source.max_angle_from_radius(a,r)==pytest.approx(expected)
    assert c.source.max_angle_from_radius(1e-14,5e-15)==pytest.approx(math.asin(.5))
    with pytest.raises(ValueError): c.source.max_angle_from_radius(1,-1)


def make_index():
    rng=np.random.default_rng(8)
    p=np.vstack([np.zeros((1,4)),rng.normal(size=(39,4))]).astype(np.float32)
    index=c.source.RandomProjectionHashIndex('fixture',p,3,5,rng,True,0)
    return p,index


def test_nested_disjoint_coverage_and_zero_vectors():
    p,index=make_index()
    e=c.Evaluator(p,1.,4.,layers=8,index=index,delta=.3,min_collisions=2)
    for q in [np.zeros(4),np.array([1.,0.,0.,0.]),np.array([.1,.2,-.3,.4])]:
        prev=set()
        codes=index.query_codes(q)
        for radius in [.01,.4,1.,2.,4.,20.]:
            union,_,rings=e.partition(q,codes,radius)
            assert prev<=union
            assert len(union)==sum(map(len,rings))
            assert union==set().union(*rings)
            assert (0 in union)==(np.linalg.norm(q)<=radius)
            prev=union
        assert len(prev)==len(p)
    # All-candidate census equals f_C, not an independently chosen denominator.
    q=np.ones(4,dtype=np.float32)
    union,_,_=e.partition(q,index.query_codes(q),e.radius)
    got=e.evaluate('approx_annular_mech',q,100,np.random.default_rng(9))
    logtruth=c.log_weighted_estimate(-c.sqdist(e.points[sorted(union)],q)/2,np.zeros(len(union)),len(p))
    assert got['log_estimate']==pytest.approx(logtruth)
    assert got['hash_id_checks']>0


def test_hamming_collision_rule_and_no_truth_interface():
    p,index=make_index()
    q=p[2]
    codes=index.query_codes(q)
    union=c.source.hash_filter_candidates(index,codes,set(range(len(p))),1,0)
    joint=c.source.hash_filter_candidates(index,codes,set(range(len(p))),3,0)
    assert joint<=union and 2 in joint
    import inspect
    assert list(inspect.signature(c.Evaluator.evaluate).parameters)==['self','method','q','budget','rng']


def test_preprocessing_does_not_fit_test_rows():
    raw=np.array([[0.,1.],[2.,3.],[100.,100.],[4.,5.]])
    split={'train':np.array([0,1]),'reference':np.array([3]),'test':np.array([2])}
    a,scaler,anchor=c.fit_transform_splits(raw,split)
    assert scaler.data_max_.tolist()==[2.,3.]
    assert (a['test']>1).all()
    assert np.allclose(a['train'][0],anchor)


def test_stable_relative_error_and_per_budget_aggregation():
    err=c.errors(-1000+math.log(1.1),-1000)
    assert err['relative_error']==pytest.approx(.1)
    assert err['truth_float_underflow']==1
    assert c.errors(-math.inf,-1000)['relative_error']==1
    frame=pd.DataFrame([dict(dataset='x',method='mc',budget=b,relative_error=v,
            absolute_error=v,time_ns=1000000,kernel_evaluations=b,zero_estimate=0,
            query_id=i) for i,(b,v) in enumerate([(32,1.),(32,3.),(64,10.)])])
    out=c.aggregate(frame).set_index('budget')
    assert out.loc[32,'relative_error']==2
    assert out.loc[64,'relative_error']==10
