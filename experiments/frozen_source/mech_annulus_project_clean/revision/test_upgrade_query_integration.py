"""Run the original evaluator bodies without unrelated MECH training imports.

Only constructor dependencies for the unused norm-angle route are stubbed.
Both evaluator methods and the production sampler are imported unchanged.
"""
import json
import sys
import types
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

core=types.ModuleType('core')
core.sqdist=lambda points,q:cdist(np.asarray(q,dtype=float)[None,:],points,'sqeuclidean')[0]
sys.modules['core']=core
common=types.ModuleType('sprint_common');common.ROOT=Path('.')
sys.modules['sprint_common']=common
scores=types.ModuleType('sprint_scores');scores.ScoreModel=None;scores.quantile_groups=None
sys.modules['sprint_scores']=scores

from sprint_evaluator import SprintEvaluator
from sprint_projection import ProjectionScore
from upgrade_evaluator import EquivalentPartitionEvaluator
from upgrade_profile import profile_current


def run():
    rng=np.random.default_rng(20260920)
    points=rng.normal(size=(720,37)).astype(np.float32)
    projection=ProjectionScore(points,points[:200])
    current=SprintEvaluator.__new__(SprintEvaluator)
    current.points=points.astype(float);current.n=len(points);current.h=2.
    current.id_array=np.arange(len(points),dtype=np.int64);current.projection=projection
    fast=EquivalentPartitionEvaluator.__new__(EquivalentPartitionEvaluator)
    fast.__dict__=current.__dict__.copy()
    cfg=dict(score='projection64',grouping='quantile',full_target=True,
             H_fraction=.25,J=8,allocation='proportional')
    checked=0
    for q in rng.normal(size=(10,37)).astype(np.float32):
        for M in [32,64,128,256,512]:
            a=current.prepare(q,M,cfg);b=fast.prepare(q,M,cfg)
            assert np.array_equal(np.sort(a['H']),np.sort(b['H']))
            assert all(np.array_equal(np.sort(x),np.sort(y)) for x,y in zip(a['cells'],b['cells']))
            for ev in [current,fast]:
                row=ev.evaluate(q,M,np.random.default_rng(9),cfg,keep_ids=True)
                assert row['actual_samples']==row['kernel_evaluations']==M
                assert row['full_dimension_distance_or_dot_evaluations']==M+64
                assert len(set(row['selected_ids']))==M
                assert np.isfinite(row['log_estimate']) and row['unassigned_ns']>=0
                assert row['H_count']==M//4
            measured=profile_current(current,q,M,np.random.default_rng(9))
            reference=current.evaluate(q,M,np.random.default_rng(9),cfg)
            assert measured['log_estimate']==reference['log_estimate']
            assert measured['selected_original_vector_copy_count']==M
            checked+=1
    print(json.dumps(dict(full_query_integration_cases=checked,all_passed=True,
        scope='Production projection/evaluator/sampler bodies; synthetic points; no performance claim'),indent=2))

if __name__=='__main__':run()
