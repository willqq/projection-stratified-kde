"""Offline Table 2A: frozen-score grouping, fixed C, proportional allocation.

This diagnostic never removes H and never uses pilot/Neyman allocation.
True query/reference distances appear only here as offline variance controls.
Do not execute concurrently with formal query-latency measurement.
"""
import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import core as c
from sprint_common import ROOT,DATASETS,load_data,save_json
from sprint_scores import ScoreModel,quantile_groups
from sprint_projection import ProjectionScore

BUDGETS=(64,128,256)
RANDOM_PARTITION_SEEDS=(0,1,2,3,4)
BASE_SEED=20260916


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _configuration(node):
    if isinstance(node,dict):
        if all(k in node for k in ('score','J','grouping')):
            return node
        for key in ('method_config','selected_config','config','final_method','method'):
            if key in node:
                result=_configuration(node[key])
                if result is not None:return result
    return None


def load_frozen_config():
    manifest_path=ROOT/'freeze_manifest.json'
    config_path=ROOT/'final_config.yaml'
    if not manifest_path.exists() or not config_path.exists():
        raise RuntimeError('Both freeze_manifest.json and final_config.yaml required before formal diagnostics')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('frozen',True) is not True:
        raise RuntimeError('Method is not frozen')
    document=yaml.safe_load(config_path.read_text())
    cfg=_configuration(document)
    if cfg is None:raise RuntimeError('No frozen score/J/grouping configuration found')
    frozen_cfg=_configuration(manifest)
    if frozen_cfg is not None and any(frozen_cfg.get(k)!=cfg.get(k) for k in cfg):
        raise RuntimeError('Manifest and final configuration disagree')
    expected=manifest.get('final_config_sha256')
    if expected is not None and expected!=sha_file(config_path):
        raise RuntimeError('Final configuration changed after freeze')
    if cfg['score'] not in ('projection64','hamming','real'):
        raise ValueError('Unsupported frozen score: '+str(cfg['score']))
    if cfg['grouping'] not in ('quantile','old'):
        raise ValueError('Unsupported frozen grouping: '+str(cfg['grouping']))
    if int(cfg['J'])<1:raise ValueError('Positive frozen J required')
    return cfg,dict(final_config_sha256=sha_file(config_path),freeze_manifest_sha256=sha_file(manifest_path))


def check_cells(cells,C):
    ids=np.concatenate(cells) if cells else np.zeros(0,dtype=np.int64)
    if not np.array_equal(np.sort(ids),np.sort(C)) or len(np.unique(ids))!=len(ids):
        raise AssertionError('Grouping must be a disjoint partition of exactly C')


def variance_for_cells(cells,scaled_kernel,budget,n):
    sizes=np.asarray([len(x) for x in cells],dtype=np.int64)
    counts=c.allocate(sizes,int(budget))
    variance=c.predicted_variance([scaled_kernel[x] for x in cells],counts,n)
    return float(variance),counts


def grouping_family(C,approximate_cells,true_d2,raw_row_id,J):
    check_cells(approximate_cells,C)
    sizes=[len(x) for x in approximate_cells]
    bounds=np.cumsum([0]+sizes)
    true_ranked=C[np.lexsort((C,true_d2[C]))]
    ordered=[true_ranked[bounds[i]:bounds[i+1]] for i in range(len(sizes))]
    family=[('approx',-1,approximate_cells),('ordered',-1,ordered)]
    for seed in RANDOM_PARTITION_SEEDS:
        rng=np.random.default_rng(np.random.SeedSequence([BASE_SEED,int(raw_row_id),int(J),seed]))
        shuffled=rng.permutation(C)
        family.append(('random',seed,[shuffled[bounds[i]:bounds[i+1]] for i in range(len(sizes))]))
    for _,_,cells in family:
        check_cells(cells,C)
        if [len(x) for x in cells]!=sizes:raise AssertionError('All groupings need identical population sizes')
    return family


def run_dataset(name,phase):
    if phase not in ('test','audit'):raise ValueError('Only frozen formal phases supported')
    cfg,provenance=load_frozen_config()
    out=ROOT/'results/fixed_c_final'/phase
    out.mkdir(parents=True,exist_ok=True)
    marker=out/(name+'_DONE.json')
    if marker.exists() or (out/(name+'_queries.csv')).exists():
        raise RuntimeError('Formal diagnostic output exists; do not overwrite '+name+'/'+phase)
    if phase=='audit':
        locked=json.loads((ROOT/'audit/locked_ids.json').read_text())
        if not locked['datasets'][name]['audit_ids']:
            save_json(out/(name+'_SKIPPED.json'),dict(dataset=name,phase=phase,reason='No unused independent audit rows',**provenance))
            return
    start=time.time()
    data=load_data(name,phase)
    points=data['points'];n=len(points);h=data['old'].h
    reference=data['arrays']['reference'];training=data['arrays']['train']
    if cfg['score']=='projection64':
        score=ProjectionScore(reference,training,ROOT/'models/projection64'/name)
    else:
        score=ScoreModel(reference,training,data['model'],data['index'],ROOT/'models/score_calibration'/name)
    save_json(out/(name+'_protocol.json'),dict(dataset=name,phase=phase,final_method=cfg,
        table2a=dict(H_count=0,pilot_count=0,allocation='minimum-one then residual-capacity proportional',
                    budgets=list(BUDGETS),random_partition_seeds=list(RANDOM_PARTITION_SEEDS),
                    seed_rule='SeedSequence([20260916, raw_row_id, J, random_partition_seed])',
                    controls='same C, identical cell populations and fixed allocations',
                    diagnostic_time_is_not_online_query_latency=True),
        offline=score.offline_record,**provenance))
    variance_rows=[];query_rows=[];stratum_rows=[]
    for qi,q in enumerate(data['arrays'][phase]):
        row_id=int(data['split'][phase][qi]);J=int(cfg['J'])
        t=time.perf_counter_ns()
        if cfg['score']=='projection64':
            C=np.arange(n,dtype=np.int64);z=score.query(q);approx=quantile_groups(C,z['d2hat'],J)
        else:
            enc=score.encode_query(q)
            union,_,old=data['old'].partition(q,enc['codes'],data['old'].radius)
            C=np.asarray(sorted(union),dtype=np.int64)
            z=score.query(q,C,codes=enc['codes'],encoded=enc,kind=cfg['score'])
            approx=([np.asarray(x,dtype=np.int64) for x in old if len(x)] if cfg['grouping']=='old'
                    else quantile_groups(C,z['d2hat'],J))
        offline_score_ns=time.perf_counter_ns()-t
        t=time.perf_counter_ns();d2=c.sqdist(points,q);logk=-d2/(2*h*h)
        logscale=float(logk.max());values=np.exp(logk-logscale)
        scale=float(np.exp(logscale));scale2=float(np.exp(2*logscale))
        true_log_mean=c.log_weighted_estimate(logk,np.zeros(n),n)
        offline_truth_ns=time.perf_counter_ns()-t
        base=dict(dataset=name,phase=phase,query_id=qi,raw_row_id=row_id,score=cfg['score'],J=J,
            candidate_size=len(C),candidate_fraction=len(C)/n,H_count=0,pilot_count=0,
            candidate_kernel_mass_recall=float(values[C].sum()/values.sum()),
            kernel_scale_log=logscale,log_truth=true_log_mean,truth=float(np.exp(true_log_mean)))
        query_rows.append(dict(base,candidate_ids=json.dumps(C.tolist()),
            offline_score_ns=offline_score_ns,offline_truth_ns=offline_truth_ns,
            offline_true_distance_evaluations=n,offline_true_kernel_evaluations=n))
        if len(C)==0:continue
        family=grouping_family(C,approx,d2,row_id,J)
        for grouping,seed,cells in family:
            for si,ids in enumerate(cells):
                v=values[ids];vvar=float(v.var(ddof=1)) if len(ids)>1 else 0.
                stratum_rows.append(dict(base,grouping=grouping,partition_seed=seed,stratum=si,
                    population=len(ids),member_ids=json.dumps(ids.tolist()),
                    scaled_kernel_mean=float(v.mean()),scaled_kernel_variance=vvar,
                    scaled_kernel_min=float(v.min()),scaled_kernel_max=float(v.max()),
                    kernel_mean=float(v.mean()*scale),kernel_variance=float(vvar*scale2),
                    kernel_min=float(v.min()*scale),kernel_max=float(v.max()*scale),
                    exact_distance_mean=float(np.sqrt(d2[ids]).mean())))
            for M in BUDGETS:
                variance,counts=variance_for_cells(cells,values,M,n)
                uniform,uniform_counts=variance_for_cells([C],values,M,n)
                expected=c.allocate([len(x) for x in approx],M)
                if not np.array_equal(counts,expected):raise AssertionError('Groupings have different fixed allocation')
                if counts.sum()!=uniform_counts.sum():raise AssertionError('Actual evaluation budget mismatch')
                variance_rows.append(dict(base,grouping=grouping,partition_seed=seed,budget=M,
                    actual_samples=int(counts.sum()),layer_sizes=json.dumps([len(x) for x in cells]),
                    allocation=json.dumps(counts.tolist()),scaled_predicted_variance=variance,
                    scaled_uniform_variance=uniform,predicted_variance=variance*scale2,
                    uniform_variance=uniform*scale2,variance_ratio=variance/uniform if uniform>0 else np.nan))
        if (qi+1)%50==0:print(json.dumps(dict(event='fixed_c_diagnostic',dataset=name,phase=phase,queries=qi+1)),flush=True)
    pd.DataFrame(query_rows).to_csv(out/(name+'_queries.csv'),index=False)
    pd.DataFrame(stratum_rows).to_csv(out/(name+'_strata.csv'),index=False)
    frame=pd.DataFrame(variance_rows);frame.to_csv(out/(name+'_variance.csv'),index=False)
    # Five random partitions are averaged within query, never counted as independent queries.
    within=frame.groupby(['dataset','phase','query_id','grouping','budget'],as_index=False).variance_ratio.mean()
    summary=within.groupby(['dataset','phase','grouping','budget'],as_index=False).agg(
        variance_ratio_median=('variance_ratio','median'),variance_ratio_mean=('variance_ratio','mean'),
        queries=('query_id','size'))
    summary.to_csv(out/(name+'_summary.csv'),index=False)
    save_json(marker,dict(dataset=name,phase=phase,queries=len(query_rows),variance_rows=len(frame),
        stratum_rows=len(stratum_rows),elapsed_seconds=time.time()-start,
        source_sha256=sha_file(__file__),**provenance))
    print(summary.to_json(orient='records'),flush=True)


def self_test():
    # Exhaustive without-replacement draws from a fixed candidate subset.
    n=8;C=np.array([0,2,3,5,7],dtype=np.int64)
    values=np.array([.2,.9,.8,.1,.4,.7,.3,.05])
    approximate=np.array([.8,.4,.6,.3,.2])
    cells=quantile_groups(C,approximate,2)
    family=grouping_family(C,cells,np.arange(n,dtype=float)**2,17,2)
    checks=[]
    for grouping,seed,groups in family:
        variance,counts=variance_for_cells(groups,values,3,n)
        options=[list(itertools.combinations(ids,int(m))) for ids,m in zip(groups,counts)]
        estimates=[]
        for selected in itertools.product(*options):
            estimates.append(sum(len(ids)/m*values[list(chosen)].sum() for ids,m,chosen in zip(groups,counts,selected))/n)
        expected=values[C].sum()/n
        assert abs(np.mean(estimates)-expected)<1e-14
        assert abs(np.var(estimates)-variance)<1e-14
        checks.append(dict(grouping=grouping,partition_seed=seed,draw_combinations=len(estimates),
            finite_population_variance_equivalence=True,conditional_mean_equivalence=True))
    all_variance,all_counts=variance_for_cells(cells,values,len(C),n)
    assert all_variance==0 and all_counts.sum()==len(C)
    # A diagnostic all-population draw is exact irrespective of grouping.
    result=dict(passed=True,enumerated_groupings=len(checks),checks=checks,full_candidate_budget_zero_variance=True,
        no_dataset_queries_read=True,source_sha256=sha_file(__file__))
    save_json(ROOT/'results/group_diagnostics/fixed_c_final_self_test.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['test','audit'])
    parser.add_argument('--dataset',choices=DATASETS+['all'],default='all')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if args.phase is None:parser.error('--phase is required unless --self-test')
    for name in (DATASETS if args.dataset=='all' else [args.dataset]):run_dataset(name,args.phase)

if __name__=='__main__':main()
