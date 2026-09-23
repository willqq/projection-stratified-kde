"""Check the portable wrapper against unmodified frozen evaluator methods.

Only the class AST is loaded, avoiding the unused MECH/DEANN import graph.
No expression inside the frozen class is rewritten. Timing is not compared.
"""
from pathlib import Path
import ast, json, math, sys, time, types
import numpy as np
from scipy.spatial.distance import cdist
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from projected_kde import ProjectedKDE
from projected_kde._frozen.sprint_estimator import estimate_strata

source = ROOT/'experiments/frozen_source/mech_annulus_project_clean/revision/sprint_evaluator.py'
tree = ast.parse(source.read_text())
cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name=='SprintEvaluator')
scope = dict(np=np, math=math, time=time, estimate_strata=estimate_strata,
             c=types.SimpleNamespace(sqdist=lambda p,q: cdist(np.asarray(q,dtype=np.float64)[None,:],p,metric='sqeuclidean')[0]))
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), scope)
cfg=dict(score='projection64',grouping='quantile',full_target=True,H_fraction=.25,J=8,allocation='proportional')
tested=0
for n in [720, 1800]:
    rng=np.random.default_rng(100+n)
    train=rng.normal(size=(80,71)).astype('float32')
    refs=rng.normal(size=(n,71)).astype('float32')
    model=ProjectedKDE(refs,train,3.0)
    old=scope['SprintEvaluator'].__new__(scope['SprintEvaluator'])
    old.points=model.points;old.n=n;old.h=model.h
    old.projection=model.projection;old.id_array=model.ids
    for qi,q in enumerate(rng.normal(size=(10,71)).astype('float32')):
        for budget in [32,64,128,256,512]:
            for single in [False,True]:
                seed=qi*100003+2
                a=model.query(q,budget,seed,single)
                b=old.evaluate(q,budget,np.random.default_rng(seed),cfg,
                               control='single' if single else None,keep_ids=True)
                assert a['log_estimate']==b['log_estimate']
                assert np.array_equal(a['selected_ids'],b['selected_ids'])
                assert np.array_equal(a['H_ids'],b['H_ids'])
                assert a['actual_samples']==b['actual_samples']==budget
                assert len(set(a['selected_ids']))==budget
                tested+=1
# Exact census is independent of auxiliary score quality.
q=train[0];a=model.query(q,n,123)
exact=np.exp(-cdist(q[None,:].astype(float),model.points,'sqeuclidean')[0]/18).mean()
assert np.isclose(a['estimate'],exact,rtol=1e-13,atol=0)
print(json.dumps(dict(passed=True,matched_frozen_evaluations=tested,
                     same_selected_ids=True,same_H=True,same_estimates=True,
                     budgets=[32,64,128,256,512],census_passed=True),indent=2))
