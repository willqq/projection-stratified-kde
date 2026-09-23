import numpy as np
import pytest
import core as c
import deann_adapter as d


def test_official_gaussian_and_full_coverage():
    rng=np.random.default_rng(8)
    p=rng.normal(size=(100,7)).astype(np.float64)
    queries=rng.normal(size=(4,7))
    exact=d.deann.NaiveKde(1.5,'gaussian');exact.fit(p)
    means,counts=exact.query(queries)
    target=np.array([np.exp(c.exact_log_mean(p,q,1.5)) for q in queries])
    np.testing.assert_allclose(means,target,rtol=1e-10,atol=1e-14)
    assert (counts==len(p)).all()
    ann,_=d.build(p)
    est,k,m,_=d.make_estimator(p,1.5,ann,100,.5,2,0)
    for q,truth in zip(queries,target):
        got=d.evaluate(est,ann,q,k,m)
        assert np.exp(got['log_estimate'])==pytest.approx(truth,rel=1e-9)
        assert got['distance_evaluations']>=100


def test_official_partial_sampling_moments():
    rng=np.random.default_rng(91)
    p=rng.normal(size=(100,7)).astype(np.float64)
    q=rng.normal(size=7)
    truth=np.exp(c.exact_log_mean(p,q,1.5))
    ann,_=d.build(p)
    values=[]
    for seed in range(200):
        est,k,m,_=d.make_estimator(p,1.5,ann,32,.25,2,seed)
        got=d.evaluate(est,ann,q,k,m)
        assert got['actual_samples']==32
        values.append(np.exp(got['log_estimate']))
    se=np.std(values,ddof=1)/np.sqrt(len(values))
    assert abs(np.mean(values)-truth)<5*se
