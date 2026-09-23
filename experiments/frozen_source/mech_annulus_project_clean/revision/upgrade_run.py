"""Bounded preregistered validation/formal runner on the frozen server setup.

Run serially on the original hardware. Original test/audit remain exposed.
Raw repetition rows retain actual work counts and include all online phases.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import core as c
import run as r
from sprint_common import load_data, save_json, DATASETS
from sprint_evaluator import SprintEvaluator
from sprint_projection import ProjectionScore
from upgrade_evaluator import EquivalentPartitionEvaluator
from upgrade_ann import ANNControlledEvaluator, conditional_variance
from upgrade_partition import profile_original, split_scores
from upgrade_profile import profile_current
import deann_adapter as da

CFG=dict(score='projection64',grouping='quantile',full_target=True,
         H_fraction=.25,J=8,allocation='proportional')


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def emit(file,row):
    file.write(json.dumps(row,default=json_default)+'\n')


def environment():
    import scipy,torch
    return dict(unix=time.time(),platform=platform.platform(),numpy=np.__version__,
        scipy=scipy.__version__,torch=torch.__version__,faiss=da.faiss.__version__,
        cpu=subprocess.check_output(['lscpu'],text=True),load_average=os.getloadavg(),
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        git_head=r.git_head(),git_status=subprocess.check_output(['git','status','--short'],cwd=r.ROOT,text=True),
        source_sha256={p.name:r.sha(p) for p in Path(__file__).parent.glob('*.py')})


def anchors(data,prior,scale_n=None):
    original=json.loads((prior/'inputs/original_final_config.json').read_text())['datasets'][data['name']]['deann']
    frozen=(json.loads((prior/'scaling_validation'/str(scale_n)/'selection.json').read_text())
        if scale_n else json.loads((prior/'DEANN_FROZEN.json').read_text())['datasets'][data['name']])
    variants={};seen={}
    for label,x in [('O',original)]+[('E' if x['selected_by']=='mean_relative_error' else 'T',x) for x in frozen['anchors']]:
        pair=(int(x['nprobe']),float(x['near_fraction']))
        if pair in seen:
            variants[seen[pair]]['aliases'].append(label)
        else:
            variants[label]=dict(nprobe=pair[0],near_fraction=pair[1],aliases=[label]);seen[pair]=label
    return variants


def construct(data,scale_n,prior,phase):
    current=SprintEvaluator(data);current.ensure_projection()
    queries=data['arrays'][phase];rawids=data['split'][phase]
    if scale_n:
        if data['name']!='cifar10_gist512':raise ValueError('Only existing GIST scale sequence')
        frozen=json.loads((prior/'scaling_ids.json').read_text())
        assert r.sha(r.DATA/r.FILES[data['name']])==frozen['raw_sha256']
        raw=r.raw_dataset(data['name'])
        ids=np.asarray(frozen['reference_order'][:scale_n],dtype=np.int64)
        refs=(2*data['scaler'].transform(raw[ids])+data['anchor']).astype(np.float32)
        current.points=np.ascontiguousarray(refs,dtype=np.float64)
        current.n=scale_n;current.id_array=np.arange(scale_n,dtype=np.int64)
        current.projection=ProjectionScore(refs,data['arrays']['train'],r.JOBROOT/'upgrade_models'/str(scale_n))
        if phase=='audit':
            rawids=np.asarray(frozen['query_ids'],dtype=np.int64)
            queries=(2*data['scaler'].transform(raw[rawids])+data['anchor']).astype(np.float32)
        assert not set(ids)&set(rawids)
    fast=EquivalentPartitionEvaluator.__new__(EquivalentPartitionEvaluator)
    fast.__dict__=current.__dict__.copy()
    return current,fast,queries,rawids


def run(args):
    if any(os.environ.get(k)!='8' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):
        raise RuntimeError('Set all BLAS/OMP thread counts to 8 before Python starts')
    if args.phase!='validation' and not (args.out.parent/'UPGRADE_FREEZE.json').exists():
        raise RuntimeError('Formal evaluation requires a validation decision and freeze')
    args.out.mkdir(parents=True,exist_ok=False)
    save_json(args.out/'START.json',environment())
    frozen_methods=None
    if args.phase!='validation':
        frozen_methods=json.loads((args.out.parent/'UPGRADE_FREEZE.json').read_text())
    datasets=['cifar10_gist512'] if args.scale else DATASETS
    budgets=([64,128,256] if args.phase=='validation' or args.scale else [32,64,128,256,512])
    interleave=np.random.default_rng(20260920)
    for name in datasets:
        if args.phase=='audit' and name=='amazon':continue
        folder=args.out/name;folder.mkdir()
        data=load_data(name,args.phase)
        current,fast,queries,rawids=construct(data,args.scale,args.prior,args.phase)
        if args.smoke:queries=queries[:2];rawids=rawids[:2]
        control=c.Evaluator(current.points,current.h,current.old.radius,8)
        variants=anchors(data,args.prior,args.scale)
        # Clone one deterministic fitted IVF so each anchor retains its own
        # nprobe while all controlled and official methods use identical tables.
        first,annoff=da.build(current.points)
        anns={};hybrids={}
        for label,v in variants.items():
            ann=copy.copy(first);ann.index=da.faiss.clone_index(first.index)
            ann.set_query_arguments(v['nprobe']);anns[label]=ann
            hybrids[label]=ANNControlledEvaluator(current.points,current.h,current.projection,ann)
        # Truth stays exclusively in the evaluation harness.
        truths=[c.exact_log_mean(current.points,q,current.h) for q in queries]
        save_json(folder/'OFFLINE.json',dict(ann=annoff,projection=current.projection.offline_record,
            n=current.n,d=current.points.shape[1],h=current.h,R=current.old.radius,
            variants=variants,query_rows=rawids,phase=args.phase,exposed_queries=args.phase!='validation'))
        with (folder/'partition_checks.jsonl').open('w') as checks:
            for qi,q in enumerate(queries):
                z=current.projection.query(q)
                for M in budgets:
                    for rep in range(3):
                        for implementation in interleave.permutation(['original','multiselect']):
                            if implementation=='original':
                                H,a,detail=profile_original(z['d2hat'],current.id_array,M//4,8)
                            else:
                                H2,b,newdetail=split_scores(z['d2hat'],current.id_array,M//4,8)
                        assert np.array_equal(np.sort(H),np.sort(H2))
                        assert all(np.array_equal(np.sort(x),np.sort(y)) for x,y in zip(a,b))
                        emit(checks,dict(query_id=qi,budget=M,repeat=rep,members_equal=True,**detail,**newdetail))
        with (folder/'current_profile.jsonl').open('w') as profile:
            for qi,q in enumerate(queries):
                reference=current.evaluate(q,128,np.random.default_rng(qi*100003),CFG)
                for rep in range(3):
                    row=profile_current(current,q,128,np.random.default_rng(qi*100003))
                    assert abs(row['log_estimate']-reference['log_estimate'])<1e-12
                    emit(profile,dict(query_id=qi,budget=128,repeat=rep,**row))
        with (folder/'raw_queries.jsonl').open('w') as rawout, (folder/'ann_variance.jsonl').open('w') as varout:
            for M in budgets:
                for seed in range(1 if args.smoke else 5):
                    official={}
                    for label,v in variants.items():
                        official[label]=da.make_estimator(current.points,current.h,anns[label],M,v['near_fraction'],v['nprobe'],seed)[:3]
                    methods=['exact','uniform_mc','exact_annular','projected_current']
                    if frozen_methods is None or frozen_methods.get('path_a',False):
                        methods.append('projected_fast')
                    methods+=['official_'+label for label in variants]
                    if not args.scale:
                        for label,v in variants.items():
                            if any(x in v['aliases'] for x in ['E','T']):
                                if frozen_methods is None or label in frozen_methods.get('path_b_anchors',[]):
                                    methods+=['ann_'+label+'_single','ann_'+label+'_multi']
                    def one(method,q,qi):
                        rng=np.random.default_rng(seed+qi*100003)
                        if method=='projected_current':return current.evaluate(q,M,rng,CFG)
                        if method=='projected_fast':return fast.evaluate(q,M,rng,CFG)
                        if method.startswith('official_'):
                            label=method.split('_')[1];est,k,m=official[label]
                            return da.evaluate(est,anns[label],q,k,m)
                        if method.startswith('ann_'):
                            _,label,mode=method.split('_');v=variants[label]
                            return hybrids[label].evaluate(q,M,rng,v['near_fraction'],v['nprobe'],mode)
                        return control.evaluate(method,q,M,rng)
                    for method in methods:one(method,queries[0],0)
                    for qi,q in enumerate(queries):
                        if seed==0 and not args.scale:
                            # Offline controls keep all observed kernels out of
                            # the online evaluators; stable scale cancels in ratios.
                            logs=-c.sqdist(current.points,q)/(2*current.h**2)
                            values=np.exp(logs-logs.max())
                            for label,v in variants.items():
                                if not any(x in v['aliases'] for x in ['E','T']):continue
                                p=[]
                                for mode in ['single','multi']:
                                    H,cells,_,_,_=hybrids[label].prepare(q,M,v['near_fraction'],v['nprobe'],mode)
                                    assert len(np.unique(np.concatenate([H]+cells)))==current.n
                                    p.append((H,cells))
                                assert np.array_equal(p[0][0],p[1][0])
                                v0=conditional_variance(values,current.n,p[0][1],M-len(p[0][0]))
                                v1=conditional_variance(values,current.n,p[1][1],M-len(p[1][0]))
                                emit(varout,dict(query_id=qi,budget=M,anchor=label,H=p[0][0],
                                    single_variance_scaled=v0,multi_variance_scaled=v1,
                                    variance_ratio=v1/v0 if v0 else None,
                                    log_kernel_scale=float(logs.max())))
                        for rep in range(3):
                            for method in interleave.permutation(methods):
                                row=one(str(method),q,qi)
                                if 'full_dimension_distance_or_dot_evaluations' not in row:
                                    row['full_dimension_distance_or_dot_evaluations']=row['distance_evaluations']
                                row.setdefault('low_dim_score_evaluations',0)
                                log_estimate=row.pop('log_estimate')
                                emit(rawout,dict(dataset=name,phase=args.phase,n=current.n,
                                    query_id=qi,raw_row_id=int(rawids[qi]),budget=M,seed=seed,
                                    repeat=rep,method=str(method),
                                    **c.errors(log_estimate,truths[qi]),**row))
                        if qi%20==0:
                            rawout.flush();varout.flush()
                            print(json.dumps(dict(dataset=name,phase=args.phase,budget=M,seed=seed,query=qi)),flush=True)
                    del official
        save_json(folder/'DONE.json',dict(unix=time.time(),queries=len(queries),budgets=budgets))
    save_json(args.out/'DONE.json',dict(unix=time.time(),scope='smoke only' if args.smoke else 'protocol run'))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['validation','test','audit'],default='validation')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--scale',type=int,choices=[1000,1800,5000,10000,20000])
    ap.add_argument('--smoke',action='store_true')
    run(ap.parse_args())
