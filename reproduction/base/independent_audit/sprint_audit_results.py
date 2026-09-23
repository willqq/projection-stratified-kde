"""Independent post-run audit. Reads raw formal CSVs only, not parent summaries.

No imports from experiment modules; no query evaluation or configuration search.
Run only after all four test and three audit dataset jobs have finished.
"""
import argparse, ast, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS=['isolet','cifar10','cifar10_gist512','amazon']
BUDGETS=[32,64,128,256,512]
SEEDS=list(range(5))
BOOTSTRAP_SEED=20260916
KEY=['method','query_id','budget','seed']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def sequence(value):
    if isinstance(value,str):return list(ast.literal_eval(value))
    if value is None or (isinstance(value,float) and np.isnan(value)):return []
    return list(value)


def independent_proportional(sizes,budget):
    sizes=np.asarray(sizes,dtype=np.int64);m=np.ones(len(sizes),dtype=np.int64)
    target=min(int(budget),int(sizes.sum()))
    while int(m.sum())<target:
        capacity=sizes-m;left=target-int(m.sum())
        add=np.minimum((left*capacity)//int(capacity.sum()),capacity)
        if int(add.sum())==0:
            ix=np.argsort(-capacity,kind='stable')[:left];add[ix]=1
        m+=add
    return m


class Audit:
    def __init__(self):self.checks=[];self.failures=[]
    def check(self,condition,name,**details):
        ok=bool(condition)
        row=dict(check=name,passed=ok,**details);self.checks.append(row)
        if not ok:self.failures.append(row)
        return ok
    def close(self,a,b,name,rtol=1e-9,atol=1e-10,**details):
        aa=np.asarray(a,dtype=float);bb=np.asarray(b,dtype=float)
        good=np.isclose(aa,bb,rtol=rtol,atol=atol,equal_nan=True)
        return self.check(np.all(good),name,mismatch_count=int(np.count_nonzero(~good)),**details)


def verify_inputs(root,checker):
    source=root/'worktree/mech_annulus_project_clean/revision'
    cfg=load_json(root/'final_config.yaml');manifest=load_json(root/'freeze_manifest.json')
    checker.check(sha(root/'final_config.yaml')==manifest['final_config_sha256'],'frozen_config_hash')
    checker.check(cfg['frozen_before_formal_evaluation'],'frozen_flag')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'worktree',text=True).strip()
    checker.check(head==cfg['source_commit'],'source_commit',actual=head,expected=cfg['source_commit'])
    for filename,expected in cfg['source_hashes'].items():
        checker.check(sha(source/filename)==expected,'frozen_source_hash',file=filename)
    for filename,expected in cfg['model_and_calibration_hashes'].items():
        checker.check(sha(root/filename)==expected,'frozen_model_calibration_hash',file=filename)
    lock=load_json(root/'audit/locked_ids.json')
    checker.check(sha(root/'audit/locked_ids.json')==cfg['audit_ids_sha256'],'audit_lock_hash')
    checker.check(lock['locked_before_validation'],'audit_locked_before_validation')
    splits={}
    for name in DATASETS:
        meta=cfg['datasets'][name];folder=root/'assets'/name
        split=load_json(folder/'split_ids.json');splits[name]=split
        checker.check(sha(folder/'split_ids.json')==meta['split_sha256'],'original_split_hash',dataset=name)
        checker.check(sha(folder/'preprocessing.npz')==meta['preprocessing_sha256'],'preprocessing_hash',dataset=name)
        expected_model=meta.get('legacy_c5_model_sha256',meta['model_sha256'])
        checker.check(sha(folder/'model.pt')==expected_model,'legacy_comparator_model_hash',dataset=name)
        sets={key:set(map(int,split[key])) for key in ('train','reference','validation','test')}
        all_ids=[int(i) for key in sets for i in split[key]]
        checker.check(len(all_ids)==len(set(all_ids)),'original_splits_disjoint',dataset=name)
        audit_ids=list(map(int,lock['datasets'][name]['audit_ids']))
        checker.check(len(audit_ids)==len(set(audit_ids))==cfg['audit_queries'][name],'audit_count_unique',dataset=name)
        checker.check(not set(audit_ids).intersection(all_ids),'audit_disjoint_all_original_splits',dataset=name)
        checker.check(len(sets['reference'])==meta['n'],'reference_count',dataset=name)
        checker.check(len(sets['test'])==200,'old_test_count',dataset=name)
        extensions=list((root/'external/deann/build').rglob('deann*.so'))
        checker.check(len(extensions)==1 and sha(extensions[0])==meta['deann']['deann_extension_sha256'],
                      'official_deann_extension_hash',dataset=name)
    proof=load_json(root/'results/estimator_checks/summary.json')
    checker.check(proof['pass'] and proof['estimator_sha256']==cfg['source_hashes']['sprint_estimator.py'] and proof['conditional_cases']>=300 and proof['max_conditional_absolute_error']<1e-12,
                  'frozen_estimator_exact_conditional_enumeration')
    return cfg,lock,splits


def check_layout(frame,timing,truth,expected_ids,cfg,name,phase,checker):
    label=dict(dataset=name,phase=phase);n=cfg['datasets'][name]['n']
    other='matched_neyman' if cfg['method_config']['allocation']=='proportional' else 'matched_proportional'
    methods={'exact','uniform_mc','exact_annular','current_approx','final','matched_single',other,'deann'}
    checker.check(set(frame.method)==methods,'method_set',**label)
    checker.check(len(frame)==35200,'estimate_row_count',actual=len(frame),**label)
    checker.check(len(timing)==105600,'timing_row_count',actual=len(timing),**label)
    checker.check(not frame.duplicated(KEY).any(),'unique_estimate_keys',**label)
    checker.check(not timing.duplicated(KEY+['repeat']).any(),'unique_timing_keys',**label)
    checker.check(len(truth)==200 and not truth.query_id.duplicated().any(),'truth_rows',**label)
    for kind,f in [('estimates',frame),('timings',timing),('truth',truth)]:
        checker.check(set(f.dataset)=={name} and set(f.phase)=={phase},'dataset_phase_labels',file_kind=kind,**label)
        ids=f.query_id.to_numpy(dtype=int)
        correct=np.all((ids>=0)&(ids<200)) and np.array_equal(f.raw_row_id.to_numpy(dtype=int),np.asarray(expected_ids,dtype=int)[ids])
        checker.check(correct,'raw_query_ids',file_kind=kind,**label)
    for method,g in frame.groupby('method'):
        if method=='exact':
            good=len(g)==200 and set(g.budget)=={0} and set(g.seed)=={0}
        else:
            counts=g.groupby(['budget','seed']).size()
            good=len(g)==5000 and set(g.budget)==set(BUDGETS) and set(g.seed)==set(SEEDS) and len(counts)==25 and (counts==200).all()
        checker.check(good,'complete_query_budget_seed_grid',method=method,**label)
    raw_counts=timing.groupby(KEY).size()
    checker.check((raw_counts==3).all() and len(raw_counts)==len(frame) and set(timing['repeat'])=={0,1,2},
                  'three_timing_repeats',**label)
    checker.check(set(map(tuple,frame[KEY].to_numpy()))==set(map(tuple,raw_counts.index.to_list())),
                  'timing_estimate_key_match',**label)
    tr=truth.set_index('query_id');expected_log=frame.query_id.map(tr.log_truth).to_numpy()
    checker.close(frame.log_truth,expected_log,'shared_log_truth',rtol=0.,atol=1e-12,**label)
    logs=frame.log_estimate.to_numpy(dtype=float)
    recomputed=np.expm1(logs-expected_log)
    checker.close(frame.signed_relative_error,recomputed,'signed_error_recomputed',**label)
    checker.close(frame.relative_error,np.abs(recomputed),'absolute_relative_error_recomputed',**label)
    checker.close(frame.truth,np.exp(expected_log),'linear_truth_recomputed',atol=1e-300,**label)
    checker.close(frame.estimate,np.exp(logs),'linear_estimate_recomputed',atol=1e-300,**label)
    checker.close(frame.absolute_error,np.exp(expected_log)*np.abs(recomputed),'absolute_error_recomputed',atol=np.maximum(1e-300,1e-12*np.exp(expected_log)),**label)
    checker.check(np.isfinite(expected_log).all() and np.isfinite(frame.relative_error).all(),'finite_truth_and_errors',**label)
    checker.close(frame.transform_ns,frame.query_id.map(tr.transform_ns),'transform_consistency',rtol=0.,atol=1e-7,**label)
    checker.close(frame.transform_inclusive_total_derived_ns,frame.time_ns+frame.transform_ns,'derived_transform_sum',rtol=0.,atol=1e-6,**label)
    checker.check(np.all(frame[frame.method=='exact'].relative_error<1e-11),'exact_numerical_error',**label)
    checker.check(np.all(frame[frame.method!='current_approx'].support_size==n),'full_population_support',**label)
    checker.close(frame[frame.method!='current_approx'].candidate_mass_recall,1.,'full_population_mass',**label)
    old=frame[frame.method=='current_approx']
    checker.close(old.candidate_mass_recall,old.query_id.map(tr.current_candidate_mass_recall),'old_candidate_mass',**label)
    checker.close(old.support_size,old.query_id.map(tr.current_candidate_size),'old_candidate_size',**label)
    checker.close(frame.candidate_union_discrepancy,frame.candidate_mass_recall-1.,'union_discrepancy',**label)


def check_timing(frame,timing,name,phase,checker):
    label=dict(dataset=name,phase=phase)
    meds=timing.groupby(KEY).median(numeric_only=True)
    ix=pd.MultiIndex.from_frame(frame[KEY]);meds=meds.reindex(ix)
    for column in frame.columns:
        if column.endswith('_ns') and column in meds:
            checker.close(frame[column],meds[column],'repeat_median_matches_summary',column=column,rtol=0.,atol=1e-6,**label)
    checker.check(np.all(timing.time_ns>0),'positive_complete_query_time',**label)
    new=timing[timing.method.isin(['final','matched_single','matched_neyman','matched_proportional'])]
    new_parts=['projection_ns','low_dim_score_ns','sorting_and_strata_ns','residual_build_ns','sampling_ns','allocation_ns','kernel_ns','aggregation_ns','unassigned_ns']
    checker.close(new.time_ns,new[new_parts].sum(axis=1),'new_raw_phase_sum',rtol=0.,atol=1e-6,**label)
    checker.check((new[new_parts]>=0).all().all(),'new_raw_phase_nonnegative',**label)
    old=timing[timing.method=='current_approx']
    old_parts=['encoding_ns','radius_ns','norm_lookup_ns','hash_lookup_ns','candidate_collect_ns','candidate_intersection_ns','dedup_ns','annulus_build_ns','sampling_ns','kernel_ns','unassigned_ns']
    checker.close(old.time_ns,old[old_parts].sum(axis=1),'old_raw_phase_sum',rtol=0.,atol=1e-6,**label)
    basic=timing[timing.method.isin(['exact','uniform_mc','exact_annular'])]
    basic_parts=['encoding_ns','radius_ns','candidate_ns','allocation_ns','kernel_ns']
    checker.check(np.all(basic[basic_parts].sum(axis=1)<=basic.time_ns),'basic_raw_phase_subtotal',**label)
    de=timing[timing.method=='deann']
    checker.close(de.time_ns,de.candidate_ns+de.kernel_ns,'deann_raw_phase_sum',rtol=0.,atol=1e-6,**label)
    checker.close(de.sampling_kernel_combined_ns,de.kernel_ns,'deann_combined_alias',rtol=0.,atol=0.,**label)


def check_counts(frame,timing,cfg,name,phase,checker):
    label=dict(dataset=name,phase=phase);n=cfg['datasets'][name]['n'];d=cfg['datasets'][name]['d']
    checker.check(np.all(frame.feature_dimension==d),'feature_dimension',**label)
    new=frame[frame.method.isin(['final','matched_single','matched_neyman','matched_proportional'])]
    H_by_query_budget={};problems=[]
    for row in new.itertuples(index=False):
        ids=sequence(row.selected_ids);H=sequence(row.H_ids);sizes=np.asarray(sequence(row.layer_sizes),dtype=int)
        a=np.asarray(sequence(row.pilot_counts),dtype=int);b=np.asarray(sequence(row.allocation),dtype=int)
        actual=min(int(row.budget),n);k=min(int(row.budget*.25),n,max(0,int(row.budget)-1))
        ok=(len(ids)==len(set(ids))==actual==row.actual_samples==row.kernel_evaluations==row.full_dimension_query_reference_distances)
        ok=ok and row.kernel_calls<=2 and sum(sequence(row.kernel_call_sizes))==actual
        ok=ok and all(0<=int(i)<n for i in ids) and len(H)==len(set(H))==k==row.H_count and ids[:k]==H
        ok=ok and len(sizes)==len(a)==len(b) and int(sizes.sum())+k==n and np.all(sizes>0)
        ok=ok and np.all((a>=0)&(a<=2)) and np.all(b>=0) and np.all(a+b<=sizes)
        ok=ok and np.all(b[sizes-a>0]>=1) and k+int(a.sum()+b.sum())==actual
        if row.method=='matched_single':ok=ok and len(sizes)==1 and sizes[0]==n-k
        else:ok=ok and len(sizes)==8 and int(sizes.max()-sizes.min())<=1
        if row.effective_allocation=='proportional':
            ok=ok and np.all(a==0) and np.array_equal(b,independent_proportional(sizes,actual-k))
        elif row.effective_allocation=='neyman':ok=ok and np.array_equal(a,np.minimum(sizes,2))
        elif row.effective_allocation=='census':ok=ok and actual==n
        else:ok=False
        key=(int(row.query_id),int(row.budget));h=tuple(H)
        if key in H_by_query_budget:ok=ok and H_by_query_budget[key]==h
        else:H_by_query_budget[key]=h
        if not ok:problems.append(dict(method=row.method,query_id=int(row.query_id),budget=int(row.budget),seed=int(row.seed)))
    checker.check(not problems,'new_ids_H_partition_allocation_and_budget',mismatch_count=len(problems),examples=problems[:5],**label)
    checker.check(np.all(new.candidate_size==n) and np.all(new.residual_size==0),'projection_C_equals_P',**label)
    checker.check(np.all(new.full_dimension_projection_inner_products==64) and np.all(new.full_dimension_encoder_inner_products==0),'projection_full_dimension_count',**label)
    checker.check(np.all(new.full_dimension_distance_or_dot_evaluations==new.kernel_evaluations+64),'new_distance_dot_total',**label)
    checker.check(np.all(new.low_dim_score_evaluations==n) and np.all(new.low_dim_score_dimension==64),'full_low_dim_scan_count',**label)
    for method in ['exact','uniform_mc','exact_annular','current_approx']:
        g=frame[frame.method==method]
        expected=n if method=='exact' else np.minimum(g.budget,g.candidate_size)
        checker.close(g.actual_samples,expected,'basic_sample_count',method=method,**label)
        checker.close(g.kernel_evaluations,g.actual_samples,'basic_kernel_count',method=method,**label)
        expected_distance=n if method in ('exact','exact_annular') else g.actual_samples
        checker.close(g.full_dimension_query_reference_distances,expected_distance,'basic_distance_count',method=method,**label)
    de=frame[frame.method=='deann'];nlist=cfg['datasets'][name]['deann']['nlist']
    checker.close(de.ivf_centroid_distance_evaluations,nlist,'deann_centroid_count',**label)
    checker.close(de.ivf_reference_distance_evaluations,de.ann_distance_evaluations-nlist,'deann_ivf_reference_count',**label)
    checker.close(de.permuted_block_samples,de.official_samples_counter-de.ann_distance_evaluations-de.ann_neighbors,'deann_block_count',**label)
    checker.close(de.overlap_corrections,np.maximum(0,de.permuted_block_samples-de.requested_residual_samples),'deann_overlap_count',**label)
    checker.close(de.kernel_evaluations,de.ann_neighbors+de.permuted_block_samples+de.overlap_corrections,'deann_kernel_count',**label)
    checker.close(de.distance_evaluations,de.ann_distance_evaluations+de.kernel_evaluations,'deann_distance_count',**label)
    checker.close(de.full_dimension_query_reference_distances,de.distance_evaluations-nlist,'deann_original_reference_count',**label)
    checker.close(de.actual_samples,de.ann_neighbors+de.requested_residual_samples,'deann_unique_contribution_count',**label)
    checker.close(de.requested_neighbors+de.requested_residual_samples,de.budget,'deann_nominal_budget',**label)
    checker.check(np.all(de.actual_samples<=de.budget),'deann_actual_budget_bound',**label)
    for method in ['final','matched_single','matched_neyman','matched_proportional']:
        g=timing[timing.method==method]
        if len(g):checker.close(g.kernel_evaluations,g.budget,'new_raw_repeat_kernel_budget',method=method,**label)
    de_t=timing[timing.method=='deann']
    checker.close(de_t.full_dimension_distance_or_dot_evaluations,
                  de_t.ivf_centroid_distance_evaluations+de_t.ivf_reference_distance_evaluations+de_t.kernel_evaluations,
                  'deann_raw_repeat_work_count',**label)



def compare_previous_baseline(root,frame,name,checker):
    folder=root/'prior_final/results/final_test'/name
    paths=[folder/'basic_queries.csv',folder/'deann_queries.csv']
    exists=all(path.exists() for path in paths)
    checker.check(exists,'prior_baseline_files_exist',dataset=name,paths=[str(path) for path in paths])
    if not exists:return [],[]
    old=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True)
    old['method']=old.method.replace({'approx_annular_mech':'current_approx'})
    names=['current_approx','uniform_mc','exact_annular','deann']
    old=old[old.method.isin(names)].copy();new=frame[frame.method.isin(names)].copy()
    checker.check(not old.duplicated(KEY).any(),'prior_baseline_unique_keys',dataset=name)
    pairs=new.merge(old,on=KEY,how='outer',suffixes=('_new','_old'),indicator=True,validate='one_to_one')
    alignment=len(pairs)==20000 and (pairs['_merge']=='both').all()
    checker.check(alignment,'prior_baseline_complete_alignment',dataset=name,rows=len(pairs),merge_counts=pairs['_merge'].value_counts().to_dict())
    if not alignment:return [],[]
    checker.close(pairs.raw_row_id_new,pairs.raw_row_id_old,'prior_baseline_same_raw_rows',rtol=0.,atol=0.,dataset=name)
    output=[];diffs=[]
    for method,g in pairs.groupby('method'):
        for metric in ['log_estimate','relative_error','actual_samples','kernel_evaluations','distance_evaluations']:
            a=g[metric+'_new'].to_numpy(dtype=float);b=g[metric+'_old'].to_numpy(dtype=float)
            tolerance=1e-12 if metric in ('log_estimate','relative_error') else 0.
            good=np.isclose(a,b,rtol=1e-10 if metric=='relative_error' else 0.,atol=tolerance,equal_nan=True)
            delta=np.zeros(len(a));finite=np.isfinite(a)&np.isfinite(b)
            delta[finite]=np.abs(a[finite]-b[finite]);delta[~finite & ~(a==b)]=np.inf
            maximum=float(np.max(delta));max_record=maximum if np.isfinite(maximum) else 'Infinity'
            checker.check(np.all(good),'prior_baseline_numeric_invariance',dataset=name,method=method,metric=metric,
                          mismatch_count=int(np.count_nonzero(~good)),max_absolute_difference=max_record)
            output.append(dict(dataset=name,method=method,metric=metric,rows=len(g),mismatch_count=int(np.count_nonzero(~good)),
                max_absolute_difference=max_record,absolute_tolerance=tolerance,relative_tolerance=1e-10 if metric=='relative_error' else 0.,
                source_basic_csv=str(paths[0].resolve()),source_deann_csv=str(paths[1].resolve()),
                source_basic_sha256=sha(paths[0]),source_deann_sha256=sha(paths[1])))
            for index in np.flatnonzero(~good):
                row=g.iloc[int(index)]
                diffs.append(dict(dataset=name,method=method,query_id=int(row.query_id),budget=int(row.budget),seed=int(row.seed),
                                  metric=metric,new_value=float(a[index]),old_value=float(b[index]),absolute_difference=float(delta[index])))
    return output,diffs


def statistics(frame,cfg,name,phase):
    summary=[];effects=[];recoveries=[]
    for (method,M),g in frame.groupby(['method','budget']):
        q=g.groupby('query_id').relative_error.mean().reindex(range(200)).to_numpy()
        summary.append(dict(dataset=name,phase=phase,method=method,budget=int(M),mean_relative_error=float(q.mean()),
            query_mean_error_median=float(np.median(q)),query_mean_error_p90=float(np.quantile(q,.9)),
            query_cluster_standard_error=float(q.std(ddof=1)/np.sqrt(200)),median_complete_query_ms=float(g.time_ns.median()/1e6),
            mean_kernel_evaluations=float(g.kernel_evaluations.mean()),mean_actual_samples=float(g.actual_samples.mean()),
            mean_full_dimension_operations=float(g.full_dimension_distance_or_dot_evaluations.mean())))
    boot=np.random.default_rng(BOOTSTRAP_SEED).integers(0,200,size=(1000,200))
    for M in BUDGETS:
        g=frame[frame.budget==M]
        pivot=g.groupby(['query_id','method']).relative_error.mean().unstack('method').reindex(range(200))
        arrays={col:pivot[col].to_numpy() for col in pivot}
        means={col:float(v.mean()) for col,v in arrays.items()}
        boots={col:v[boot].mean(axis=1) for col,v in arrays.items()}
        den=means['uniform_mc']-means['exact_annular'];valid=den>max(1e-8,.01*means['uniform_mc'])
        rec=(means['uniform_mc']-means['final'])/den if valid else np.nan
        bden=boots['uniform_mc']-boots['exact_annular']
        good=bden>np.maximum(1e-8,.01*boots['uniform_mc'])
        brec=(boots['uniform_mc'][good]-boots['final'][good])/bden[good]
        ci=np.quantile(brec,[.025,.975]) if len(brec) else [np.nan,np.nan]
        recoveries.append(dict(dataset=name,phase=phase,budget=M,MC=means['uniform_mc'],exact_annular=means['exact_annular'],
            final=means['final'],recovery=rec,recovery_ci_low=float(ci[0]),recovery_ci_high=float(ci[1]),
            stable_bootstrap_denominators=int(good.sum()),relative_error_reduction_MC=1-means['final']/means['uniform_mc']))
        comparisons=[('final',x) for x in ['uniform_mc','current_approx','deann','matched_single']]
        if 'matched_neyman' in arrays:comparisons.append(('matched_neyman','final'))
        else:comparisons.append(('final','matched_proportional'))
        for a,b in comparisons:
            diff=boots[a]-boots[b];relative=1-boots[a]/boots[b]
            ci=np.quantile(diff,[.025,.975]);rci=np.quantile(relative,[.025,.975])
            effects.append(dict(dataset=name,phase=phase,budget=M,method_a=a,method_b=b,
                paired_MARE_difference=means[a]-means[b],difference_ci_low=float(ci[0]),difference_ci_high=float(ci[1]),
                relative_error_reduction=1-means[a]/means[b],reduction_ci_low=float(rci[0]),reduction_ci_high=float(rci[1]),
                query_clusters=200,sampling_seeds_per_cluster=5,bootstrap_resamples=1000,bootstrap_seed=BOOTSTRAP_SEED))
    return summary,effects,recoveries


def main(root):
    start=time.time();out=root/'results/independent_audit';out.mkdir(parents=True,exist_ok=True)
    checker=Audit();summary=[];effects=[];recoveries=[];baseline_summary=[];baseline_differences=[];row_count=timing_count=0
    source_hash=sha(Path(__file__))
    try:
        cfg,lock,splits=verify_inputs(root,checker)
        expected=[('test',name) for name in DATASETS]+[('audit',name) for name in DATASETS if cfg['audit_queries'][name]]
        absent=[str(root/'results/formal'/phase/name/'DONE.json') for phase,name in expected
                if not (root/'results/formal'/phase/name/'DONE.json').exists()]
        checker.check(not absent,'all_formal_jobs_complete',missing=absent)
        if absent:raise RuntimeError('Formal runs are incomplete; do not audit partial results')
        for phase,name in expected:
            folder=root/'results/formal'/phase/name
            done=load_json(folder/'DONE.json')
            checker.check(done['source_commit']==cfg['source_commit'] and done['freeze_sha256']==sha(root/'final_config.yaml'),'job_freeze_provenance',dataset=name,phase=phase)
            checker.check(done['completed']>=cfg['freeze_time_unix'],'formal_after_freeze',dataset=name,phase=phase)
            frames=[pd.read_csv(folder/'basic_queries.csv'),pd.read_csv(folder/'deann_queries.csv')]
            times=[pd.read_csv(folder/'basic_timings.csv'),pd.read_csv(folder/'deann_timings.csv')]
            frame=pd.concat(frames,ignore_index=True);timing=pd.concat(times,ignore_index=True);truth=pd.read_csv(folder/'query_truth.csv')
            ids=splits[name]['test'] if phase=='test' else lock['datasets'][name]['audit_ids']
            check_layout(frame,timing,truth,ids,cfg,name,phase,checker)
            check_timing(frame,timing,name,phase,checker)
            check_counts(frame,timing,cfg,name,phase,checker)
            if phase=="test":
                bs,bd=compare_previous_baseline(root,frame,name,checker);baseline_summary.extend(bs);baseline_differences.extend(bd)
            s,e,rr=statistics(frame,cfg,name,phase);summary.extend(s);effects.extend(e);recoveries.extend(rr)
            row_count+=len(frame);timing_count+=len(timing)
            print(json.dumps(dict(event='independent_audit',dataset=name,phase=phase,rows=len(frame),timings=len(timing),failures=len(checker.failures))),flush=True)
            del frames,times,frame,timing,truth
        checker.check(row_count==246400,'all_estimate_count',actual=row_count)
        checker.check(timing_count==739200,'all_timing_count',actual=timing_count)
        # Recheck every frozen artifact after raw-file audit/statistics, not just at entry.
        verify_inputs(root,checker)
    except Exception as error:
        checker.check(False,'unhandled_audit_exception',error=repr(error))
        import traceback
        (out/'exception.txt').write_text(traceback.format_exc())
    pd.DataFrame(baseline_summary).to_csv(out/'baseline_invariance_summary.csv',index=False)
    pd.DataFrame(baseline_differences,columns=['dataset','method','query_id','budget','seed','metric','new_value','old_value','absolute_difference']).to_csv(out/'baseline_invariance_differences.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'independent_summary.csv',index=False)
    pd.DataFrame(effects).to_csv(out/'independent_paired_bootstrap.csv',index=False)
    pd.DataFrame(recoveries).to_csv(out/'independent_recovery.csv',index=False)
    if summary:
        table=pd.DataFrame(summary);table[table.budget.isin([0,128])].to_csv(out/'independent_M128.csv',index=False)
    (out/'checks.json').write_text(json.dumps(checker.checks,indent=2)+'\n')
    (out/'anomalies.json').write_text(json.dumps(checker.failures,indent=2)+'\n')
    report=dict(PASS=not checker.failures,checks=len(checker.checks),failures=len(checker.failures),
        estimate_rows=row_count,timing_rows=timing_count,bootstrap_seed=BOOTSTRAP_SEED,bootstrap_resamples=1000,
        resampling_unit='query; five sampling seeds averaged within each query',
        source_sha256=source_hash,started_unix=start,completed_unix=time.time(),
        input_scope='raw formal CSVs, frozen source/provenance, split/audit IDs; no parent aggregate tables or query reevaluation',
        limitations=['CSV records do not contain complete stratum memberships or per-item weights. Full coverage and coefficient semantics additionally rely on the independently reviewed frozen source and exact conditional-enumeration tests.',
                     'Median phase times are not assumed additive; phase accounting is checked per raw timing repeat.',
                     'Operation counts for DEANN scientific estimates use repeat zero; raw timing repeats retain their individual work counts.'])
    (out/'PASS.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# Independent audit', '', 'PASS' if report['PASS'] else 'FAIL', '',
          f"Checks: {report['checks']}; failures: {report['failures']}.",
          f'Estimate rows: {row_count}; raw timing rows: {timing_count}.',
          'All statistical summaries and 1,000 paired bootstrap replicates were computed independently from raw per-query CSVs.',
          'Each bootstrap unit is one query with its five seed errors averaged first. No parent aggregate summary was read.',
          '', 'See independent_M128.csv, independent_recovery.csv, independent_paired_bootstrap.csv, checks.json and anomalies.json.']
    (out/'INDEPENDENT_AUDIT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if checker.failures:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/icassp_sprint_20260916_0103'))
    main(parser.parse_args().root)
