"""Frozen formal old-test/audit evaluation. No selection entry point."""
import argparse,hashlib,json,math,time,subprocess
from pathlib import Path
import numpy as np,pandas as pd
import core as c,run as r
from sprint_common import ROOT,DATASETS,BUDGETS,SEEDS,load_data,save_json
from sprint_evaluator import SprintEvaluator
import deann_adapter as da


def verify_freeze():
    f=json.loads((ROOT/'final_config.yaml').read_text());m=json.loads((ROOT/'freeze_manifest.json').read_text())
    assert r.sha(ROOT/'final_config.yaml')==m['final_config_sha256']
    assert f['frozen_before_formal_evaluation'] and r.git_head()==f['source_commit']
    for filename,expected in f['source_hashes'].items():assert r.sha(Path(__file__).parent/filename)==expected,filename
    assert r.sha(ROOT/'audit/locked_ids.json')==f['audit_ids_sha256']
    return f


def normalize_counts(z,n,d,method):
    z=dict(z)
    if method in ('uniform_mc','exact_annular','exact','current_approx'):
        z['full_dimension_query_reference_distances']=int(z['distance_evaluations'])
        z['full_dimension_encoder_inner_products']=60 if method=='current_approx' else 0
        z['full_dimension_projection_inner_products']=0
        z['full_dimension_distance_or_dot_evaluations']=z['full_dimension_query_reference_distances']+z['full_dimension_encoder_inner_products']
        z['low_dim_score_evaluations']=0;z['low_dim_score_dimension']=0
    elif method=='deann':
        centers=min(32,n//40);z['ivf_centroid_distance_evaluations']=centers
        z['ivf_reference_distance_evaluations']=int(z['ann_distance_evaluations'])-centers
        z['full_dimension_query_reference_distances']=int(z['distance_evaluations'])-centers
        z['full_dimension_distance_or_dot_evaluations']=int(z['distance_evaluations'])
        z['full_dimension_encoder_inner_products']=0;z['full_dimension_projection_inner_products']=0
        z['low_dim_score_evaluations']=0;z['low_dim_score_dimension']=0
        z['sampling_kernel_combined_ns']=z['kernel_ns']
    z['feature_dimension']=d
    return z


def main(name,phase):
    freeze=verify_freeze();cfg=freeze['method_config'];data=load_data(name,phase)
    e=SprintEvaluator(data)
    if cfg['score']=='projection64':e.ensure_projection()
    control=c.Evaluator(data['arrays']['reference'],e.h,e.old.radius,8);control.points=e.points
    assert np.shares_memory(control.points,e.points)
    queries=data['arrays'][phase];rawids=data['split'][phase]
    assert len(queries)==200
    out=ROOT/'results/formal'/phase/name;out.mkdir(parents=True,exist_ok=False)
    save_json(out/'environment_start.json',dict(time_unix=time.time(),source_commit=r.git_head(),processes=subprocess.check_output(['ps','-eo','pid,pcpu,comm,args'],text=True)))
    raw=r.raw_dataset(name)
    truth=[];logks=[];transforms=[];candidate_mass=[];candidate_counts=[];truth_times=[]
    for qi,q in enumerate(queries):
        t=time.perf_counter_ns();logs=-c.sqdist(e.points,q)/(2*e.h**2);logks.append(logs);truth.append(float(c.logsumexp(logs)-math.log(e.n)));truth_times.append(time.perf_counter_ns()-t)
        codes=e.old.index.query_codes(q);C,_,_=e.old.partition(q,codes,e.old.radius);ids=np.array(sorted(C),dtype=np.int64)
        v=np.exp(logs-logs.max());candidate_mass.append(float(v[ids].sum()/v.sum()));candidate_counts.append(len(ids))
        times=[]
        for rep in range(3):
            t=time.perf_counter_ns();qq=(2*data['scaler'].transform(raw[rawids[qi]][None,:])+data['anchor']).astype(np.float32)[0];times.append(time.perf_counter_ns()-t);assert np.array_equal(qq,q)
        transforms.append(float(np.median(times)))
    del raw
    save_json(out/'offline.json',dict(legacy_comparison_setup=data['offline'],final_projection=e.projection.offline_record if e.projection is not None else None))
    pd.DataFrame([dict(dataset=name,phase=phase,query_id=i,raw_row_id=int(rawids[i]),truth=float(np.exp(truth[i])),log_truth=truth[i],transform_ns=transforms[i],truth_time_ns=truth_times[i],current_candidate_mass_recall=candidate_mass[i],current_candidate_size=candidate_counts[i]) for i in range(200)]).to_csv(out/'query_truth.csv',index=False)
    methods=['exact','uniform_mc','exact_annular','current_approx','final','matched_single']
    # Reuse the final rows as their identical allocation control; evaluate only the other allocation.
    additional='matched_neyman' if cfg['allocation']=='proportional' else 'matched_proportional';methods.append(additional)
    def evaluate(method,q,budget,seed,qi):
        rng=np.random.default_rng(seed+qi*100003)
        if method=='current_approx':z=e.old.evaluate('approx_annular_mech',q,budget,rng)
        elif method in ('exact','uniform_mc','exact_annular'):z=control.evaluate(method,q,max(1,budget),rng)
        else:
            ctrl=None if method=='final' else {'matched_single':'single','matched_neyman':'multi_neyman','matched_proportional':'multi_proportional'}[method]
            z=e.evaluate(q,budget,rng,cfg,control=ctrl,keep_ids=True)
        return normalize_counts(z,e.n,e.points.shape[1],method)
    for method in methods:evaluate(method,queries[0],32,987,0)
    rows=[];timings=[];h_check={}
    for qi,q in enumerate(queries):
        rotate=qi%len(methods)
        for method in methods[rotate:]+methods[:rotate]:
            for budget in ([0] if method=='exact' else BUDGETS):
                for seed in ([0] if method=='exact' else SEEDS):
                    reps=[evaluate(method,q,budget,seed,qi) for rep in range(3)]
                    z=reps[0].copy()
                    for rr in reps[1:]:assert rr['log_estimate']==z['log_estimate']
                    if 'selected_ids' in z:
                        ids=z['selected_ids'];assert len(ids)==len(set(ids))==min(budget,e.n)
                        key=(qi,budget);H=tuple(z['H_ids'])
                        if key in h_check:assert h_check[key]==H
                        else:h_check[key]=H
                    for k in list(z):
                        if k.endswith('_ns'):z[k]=float(np.median([rr[k] for rr in reps]))
                    z.update(c.errors(z.pop('log_estimate'),truth[qi]));z['transform_ns']=transforms[qi]
                    z['transform_inclusive_total_derived_ns']=z['time_ns']+transforms[qi]
                    z['candidate_mass_recall']=candidate_mass[qi] if method=='current_approx' else 1.
                    z['candidate_union_discrepancy']=z['candidate_mass_recall']-1
                    z['support_size']=e.n if method!='current_approx' else candidate_counts[qi]
                    z.pop('candidate_ids',None)
                    if 'H_ids' in z:
                        v=np.exp(logks[qi]-logks[qi].max());z['H_kernel_mass_recall']=float(v[z['H_ids']].sum()/v.sum())
                    base=dict(dataset=name,phase=phase,method=method,query_id=qi,raw_row_id=int(rawids[qi]),budget=budget,seed=seed)
                    rows.append(dict(base,**z))
                    for rep,rr in enumerate(reps):
                        timings.append(dict(base,repeat=rep,transform_ns=transforms[qi],**{k:v for k,v in rr.items() if k.endswith('_ns') or k in ('kernel_evaluations','actual_samples','full_dimension_distance_or_dot_evaluations','low_dim_score_evaluations','hash_bucket_visits','retrieved_posting_ids')}))
        if (qi+1)%20==0:print(json.dumps(dict(event='formal_basic',dataset=name,phase=phase,queries=qi+1)),flush=True)
    pd.DataFrame(rows).to_csv(out/'basic_queries.csv',index=False);pd.DataFrame(timings).to_csv(out/'basic_timings.csv',index=False)
    alias='matched_proportional' if cfg['allocation']=='proportional' else 'matched_neyman'
    save_json(out/'control_alias.json',dict(alias=alias,source_method='final',reason='Same frozen H, grouping and allocation; reuse identical observations rather than rerun.'))
    del rows,timings
    ann,offline=da.build(e.points);dcfg=freeze['datasets'][name]['deann'];fits=[];rows=[];timings=[]
    original_query=ann.query
    def record_neighbors(q,k):
        distances,ids,counters=original_query(q,k);ann.recorded_neighbor_ids=ids[ids>=0].copy();return distances,ids,counters
    ann.query=record_neighbors
    for budget in BUDGETS:
        for seed in SEEDS:
            est,k,m,fit=da.make_estimator(e.points,e.h,ann,budget,dcfg['near_fraction'],dcfg['nprobe'],seed);fits.append(dict(budget=budget,seed=seed,permutation_fit_seconds=fit))
            da.evaluate(est,ann,queries[0],k,m)
            for qi,q in enumerate(queries):
                reps=[];neighbors=None
                for rep in range(3):
                    zz=normalize_counts(da.evaluate(est,ann,q,k,m),e.n,e.points.shape[1],'deann');reps.append(zz)
                    if rep==0:neighbors=ann.recorded_neighbor_ids.copy()
                    timings.append(dict(dataset=name,phase=phase,method='deann',query_id=qi,raw_row_id=int(rawids[qi]),budget=budget,seed=seed,repeat=rep,transform_ns=transforms[qi],**{kk:vv for kk,vv in zz.items() if kk.endswith('_ns') or kk in ('kernel_evaluations','actual_samples','full_dimension_distance_or_dot_evaluations','ivf_centroid_distance_evaluations','ivf_reference_distance_evaluations','overlap_corrections')}))
                z=reps[0].copy()
                for kk in list(z):
                    if kk.endswith('_ns'):z[kk]=float(np.median([rr[kk] for rr in reps]))
                z.update(c.errors(z.pop('log_estimate'),truth[qi]));z['transform_ns']=transforms[qi];z['transform_inclusive_total_derived_ns']=z['time_ns']+transforms[qi]
                z['candidate_mass_recall']=1.;z['candidate_union_discrepancy']=0.;z['support_size']=e.n
                v=np.exp(logks[qi]-logks[qi].max());z['retrieved_neighbor_mass_recall']=float(v[neighbors].sum()/v.sum())
                rows.append(dict(dataset=name,phase=phase,method='deann',query_id=qi,raw_row_id=int(rawids[qi]),budget=budget,seed=seed,**z))
        print(json.dumps(dict(event='formal_deann',dataset=name,phase=phase,budget=budget)),flush=True)
    pd.DataFrame(rows).to_csv(out/'deann_queries.csv',index=False);pd.DataFrame(timings).to_csv(out/'deann_timings.csv',index=False);pd.DataFrame(fits).to_csv(out/'deann_fits.csv',index=False)
    save_json(out/'deann_offline.json',offline)
    save_json(out/'DONE.json',dict(source_commit=r.git_head(),freeze_sha256=r.sha(ROOT/'final_config.yaml'),completed=time.time(),query_count=200,budgets=BUDGETS,seeds=SEEDS))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=DATASETS);ap.add_argument('--phase',required=True,choices=['test','audit']);a=ap.parse_args();main(a.dataset,a.phase)
