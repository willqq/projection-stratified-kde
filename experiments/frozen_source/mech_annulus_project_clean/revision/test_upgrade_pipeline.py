"""Synthetic plumbing test. Temporary fake-ANN rows are never experiment data.

Real projection/sampling/profile and analysis bodies run with a lightweight ANN
double. This cannot validate FAISS/DEANN performance or substitute for the server.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def main():
    folder=Path(__file__).parent
    # Import actual core function/class AST bodies without unused MECH/sklearn
    # imports. The dataset loader is explicitly replaced by a synthetic fixture.
    tree=ast.parse((folder/'core.py').read_text())
    body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
    core=types.ModuleType('core')
    core.__dict__.update(np=np,math=math,time=time,cdist=cdist,logsumexp=logsumexp)
    exec(compile(ast.Module(body=body,type_ignores=[]),'core.py','exec'),core.__dict__)
    sys.modules['core']=core
    fake_run=types.ModuleType('run');sys.modules['run']=fake_run
    common=types.ModuleType('sprint_common');sys.modules['sprint_common']=common
    scores=types.ModuleType('sprint_scores');scores.ScoreModel=None;scores.quantile_groups=None
    sys.modules['sprint_scores']=scores
    common.ROOT=Path('.');common.DATASETS=['synthetic_fixture']
    common.load_data=lambda *args:None
    common.save_json=lambda path,value:Path(path).write_text(json.dumps(value,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    adapter=types.ModuleType('deann_adapter');sys.modules['deann_adapter']=adapter
    adapter.faiss=types.SimpleNamespace(clone_index=lambda index:index)

    from sprint_projection import ProjectionScore
    from sprint_evaluator import SprintEvaluator
    from upgrade_evaluator import EquivalentPartitionEvaluator
    from upgrade_ann import ANNControlledEvaluator

    class ANNDouble:
        def __init__(self,points):self.points=points;self.index=self
        def set_query_arguments(self,nprobe):self.nprobe=nprobe
        def query(self,q,k):
            # Mock retrieval is exact here solely to exercise API contracts.
            d2=cdist(np.asarray(q)[None,:],self.points,'sqeuclidean')[0]
            ids=np.argsort(d2)[:k]
            self.last=dict(ann_distance_evaluations=len(self.points),neighbors_returned=len(ids))
            return np.sqrt(d2[ids]),ids,np.array([len(self.points)+len(ids)])
    adapter.build=lambda points:(ANNDouble(points),{'scope':'synthetic_ANN_double'})
    def make(points,h,ann,M,fraction,nprobe,seed):
        ev=ANNControlledEvaluator(points,h,None,ann)
        return (types.SimpleNamespace(ev=ev,rng=np.random.default_rng(seed),fraction=fraction,nprobe=nprobe),int(M*fraction),M-int(M*fraction),0.)
    adapter.make_estimator=make
    adapter.evaluate=lambda est,ann,q,k,m:est.ev.evaluate(q,k+m,est.rng,est.fraction,est.nprobe,'single')

    import upgrade_run as runner
    from upgrade_analyze import analyze
    rng=np.random.default_rng(98765)
    points=rng.normal(size=(720,11)).astype(np.float32)
    queries=rng.normal(size=(2,11)).astype(np.float32)
    projection=ProjectionScore(points,points[:80])
    current=SprintEvaluator.__new__(SprintEvaluator)
    current.points=points.astype(float);current.n=len(points);current.h=2.
    current.old=types.SimpleNamespace(radius=4.)
    current.id_array=np.arange(len(points));current.projection=projection
    fast=EquivalentPartitionEvaluator.__new__(EquivalentPartitionEvaluator)
    fast.__dict__=current.__dict__.copy()
    runner.environment=lambda:{'scope':'synthetic_pipeline_fixture'}
    runner.load_data=lambda name,phase:{'name':name}
    runner.construct=lambda *args:(current,fast,queries,np.arange(2))
    runner.anchors=lambda *args:{'E':dict(nprobe=2,near_fraction=.5,aliases=['E']),
                                  'T':dict(nprobe=1,near_fraction=.125,aliases=['T'])}
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='8'
    with tempfile.TemporaryDirectory(prefix='icassp_synthetic_plumbing_') as temp:
        root=Path(temp)
        args=types.SimpleNamespace(phase='validation',out=root/'run',prior=root,scale=None,smoke=False)
        with contextlib.redirect_stdout(io.StringIO()):runner.run(args)
        raw=pd.read_json(root/'run/synthetic_fixture/raw_queries.jsonl',lines=True)
        assert len(raw)==3*5*2*3*11
        assert raw[['relative_error','time_ns','kernel_evaluations','full_dimension_distance_or_dot_evaluations']].notna().all().all()
        assert all(raw[raw.method!='exact'].actual_samples==raw[raw.method!='exact'].budget)
        with contextlib.redirect_stdout(io.StringIO()):analyze([root/'run'],root/'analysis')
        summary=pd.read_csv(root/'analysis/summary.csv')
        pairs=pd.read_csv(root/'analysis/ann_H_paired_bootstrap.csv')
        assert len(summary)==33 and len(pairs)==6
        assert set(pairs.queries)=={2}
        report=dict(raw_rows=len(raw),methods=11,budgets=3,sampling_seeds=5,
            timed_repetitions=3,summary_rows=len(summary),matched_H_comparisons=len(pairs),
            all_passed=True,scope='Synthetic pipeline plumbing with ANN double; temporary rows destroyed; no real experiment claim')
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
