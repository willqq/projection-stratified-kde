"""Independent saved-result checks. Run remotely after all frozen jobs finish."""
from pathlib import Path
import json, hashlib, math
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import run as r

b=r.JOBROOT
out=b/'results/audit_v1';out.mkdir(exist_ok=True)
manifest=json.loads((b/'run_manifest.json').read_text())
checks=[]; details={}
def check(label, condition):
    checks.append({'check':label,'passed':bool(condition)})
    if not condition:raise AssertionError(label)
def close(a,z,rtol=1e-9,atol=1e-13):return np.allclose(a,z,rtol=rtol,atol=atol,equal_nan=True)
for name in manifest['datasets']:
    meta=json.loads((b/'assets'/name/'metadata.json').read_text())
    split=json.loads((b/'assets'/name/'split_ids.json').read_text())
    ids=sum(split.values(),[])
    check(name+' disjoint split IDs', len(set(ids))==len(ids))
    check(name+' split hash',r.sha(b/'assets'/name/'split_ids.json')==meta['split_sha256'])
    check(name+' model hash',r.sha(b/'assets'/name/'model.pt')==meta['model_sha256'])
    check(name+' data hash',r.sha(r.DATA/r.FILES[name])==meta['data_sha256'])
    raw=r.raw_dataset(name)
    arrays,_,_=r.c.fit_transform_splits(raw,{k:np.array(v) for k,v in split.items()})
    points=arrays['reference'].astype(np.float64); n=len(points)
    basic=pd.read_csv(b/'results/basic_v1'/f'{name}_queries.csv')
    deann=pd.read_csv(b/'results/deann_v1'/f'{name}_deann_queries.csv')
    check(name+' basic row count',len(basic)==15200)
    check(name+' DEANN row count',len(deann)==5000)
    frame=pd.concat([basic,deann],ignore_index=True)
    keys=['query_id','method','budget','seed']
    check(name+' unique result keys',not frame.duplicated(keys).any())
    check(name+' test identity',np.array_equal(frame.raw_row_id,np.array(split['test'])[frame.query_id]))
    check(name+' common truth per query',frame.groupby('query_id').log_truth.nunique().max()==1)
    for method,part in frame.groupby('method'):
        expected=200 if method=='exact' else 5000
        check(name+' grid '+method,len(part)==expected and part.query_id.nunique()==200)
        if method!='exact':
            check(name+' frozen budget/seed grid '+method,set(part.budget)==set(manifest['budgets']) and set(part.seed)==set(manifest['sampling_seeds']) and (part.groupby(['budget','seed']).size()==200).all())
    # An alternate direct squared-difference truth calculation on fixed rows.
    for qi in [0,99,199]:
        v=-np.sum((points-arrays['test'][qi].astype(np.float64))**2,axis=1)/(2*meta['bandwidth']**2)
        truth=float(logsumexp(v)-math.log(n))
        saved=frame.loc[frame.query_id==qi,'log_truth'].iloc[0]
        check(name+f' independent truth {qi}',close(saved,truth,rtol=1e-10))
    rel=np.abs(np.expm1(frame.log_estimate-frame.log_truth))
    check(name+' relative error recomputation',close(rel,frame.relative_error))
    check(name+' absolute error recomputation',close(np.abs(frame.estimate-frame.truth),frame.absolute_error))
    check(name+' finite errors',np.isfinite(frame[['relative_error','absolute_error','time_ns','log_truth']]).all().all())
    check(name+' positive total latency',(frame.time_ns>0).all())
    check(name+' exact target',(frame.loc[frame.method=='exact','relative_error']==0).all())
    for row in basic.itertuples():
        sizes=np.array(json.loads(row.layer_sizes)); alloc=np.array(json.loads(row.allocation))
        if row.method!='exact':
            if not(len(sizes)==len(alloc) and np.all(alloc>=1) and np.all(alloc<=sizes) and alloc.sum()==min(row.budget,row.candidate_size) and sizes.sum()==row.candidate_size):raise AssertionError('capacity '+name)
    check(name+' all allocations valid',True)
    for prefix,suffix,result in [('basic_v1','',basic),('deann_v1','_deann',deann)]:
        times=pd.read_csv(b/'results'/prefix/f'{name}{suffix}_timings.csv')
        check(name+prefix+' timing rows',len(times)==len(result)*3)
        check(name+prefix+' unique repeats',not times.duplicated(keys+['repeat']).any())
        check(name+prefix+' nonnegative stages',(times.filter(regex='_ns$')>=0).all().all())
        med=times.groupby(keys).time_ns.median().sort_index()
        check(name+prefix+' median repeat calculation',close(med,result.set_index(keys).time_ns.sort_index(),rtol=0,atol=0))
        saved=pd.read_csv(b/'results'/prefix/f'{name}{suffix}_summary.csv').sort_values(['method','budget'])
        recomputed=r.c.aggregate(result).sort_values(['method','budget'])
        check(name+prefix+' independent budget aggregation',close(saved.relative_error,recomputed.relative_error))
    cfg=json.loads((b/'results/deann_v1'/f'{name}_deann_config.json').read_text())
    val=pd.read_csv(b/'results/deann_v1'/f'{name}_deann_validation.csv')
    selected=val.groupby(['nprobe','near_fraction'],as_index=False).agg(relative_error=('relative_error','mean'),time_ns=('time_ns','mean')).sort_values(['relative_error','time_ns','nprobe','near_fraction']).iloc[0]
    check(name+' DEANN validation selection',cfg['nprobe']==selected.nprobe and cfg['near_fraction']==selected.near_fraction)
    check(name+' DEANN config preceding test file',(b/'results/deann_v1'/f'{name}_deann_config.json').stat().st_mtime < (b/'results/deann_v1'/f'{name}_deann_queries.csv').stat().st_mtime)
    check(name+' DEANN requested split',(deann.requested_neighbors+deann.requested_residual_samples==deann.budget).all())
    check(name+' DEANN sample counters',(deann.permuted_block_samples>=deann.requested_residual_samples).all() and (deann.kernel_evaluations==deann.ann_neighbors+deann.permuted_block_samples+deann.overlap_corrections).all())
    details[name]={'basic_rows':len(basic),'deann_rows':len(deann),'zero_estimates':int(frame.zero_estimate.sum()),'truth_underflows':int(frame.truth_float_underflow.sum()),'min_log_truth':float(frame.log_truth.min()),'max_log_truth':float(frame.log_truth.max()),'raw_shape':list(raw.shape),'reference_n':n}
    if name in manifest['mechanism']['datasets']:
        folder=b/'results/mechanism_v1'
        draws=pd.read_csv(folder/f'{name}_mechanism_draws.csv')
        sm=pd.read_csv(folder/f'{name}_mechanism_summary.csv')
        dc=pd.read_csv(folder/f'{name}_decomposition.csv')
        parts=json.loads((folder/f'{name}_partitions.json').read_text())
        check(name+' mechanism grid',len(draws)==96000 and len(sm)==480 and len(dc)==20 and len(parts)==480)
        mk=['query_id','budget','grouping','partition_seed']
        check(name+' mechanism unique draw keys',not draws.duplicated(mk+['repeat']).any())
        emp=draws.groupby(mk).estimate_scaled.agg(['mean','var']).sort_index()
        saved=sm.set_index(mk).sort_index()
        check(name+' empirical mean from draws',close(emp['mean'],saved.mean_scaled))
        check(name+' empirical variance from draws',close(emp['var'],saved.empirical_variance_scaled))
        candidate_sets={};allocation_map={};vectors={}
        for qi in range(20):
            squared=np.sum((points-arrays['test'][qi].astype(np.float64))**2,axis=1)
            logk=-squared/(2*meta['bandwidth']**2);vals=np.exp(logk-logk.max());vectors[qi]=vals
            drow=dc.loc[dc.query_id==qi].iloc[0]
            check(name+f' decomposition full {qi}',close(vals.sum()/n,drow.full_scaled))
            check(name+f' decomposition truncated {qi}',close(vals[squared<=meta['radius']**2].sum()/n,drow.truncated_scaled))
        for part in parts:
            qi=part['query_id']; budget=part['budget'];kind=part['grouping']; ps=part['partition_seed']
            cells=part['cell_ids'];alloc=part['allocation']; flat=sum(cells,[]);C=set(flat)
            if len(C)!=len(flat):raise AssertionError('overlapping strata')
            if qi not in candidate_sets:candidate_sets[qi]=C
            if C!=candidate_sets[qi]:raise AssertionError('candidate union changed')
            if kind!='uniform_on_C':
                key=(qi,budget)
                pattern=([len(x) for x in cells],alloc)
                if key not in allocation_map:allocation_map[key]=pattern
                if pattern!=allocation_map[key]:raise AssertionError('size or allocation changed')
            values=vectors[qi]; pred=0.
            for cell,m in zip(cells,alloc):
                N=len(cell)
                if not(1<=m<=N):raise AssertionError('bad mechanism capacity')
                if N>1:
                    mean=np.mean(values[cell]);S2=sum((values[cell]-mean)**2)/(N-1)
                    pred+=N*N*(1-m/N)*S2/m/n**2
            savedrow=saved.loc[(qi,budget,kind,ps)]
            if not close(pred,savedrow.predicted_variance_scaled):raise AssertionError('FPC mismatch')
            if not close(values[list(C)].sum()/n,savedrow.candidate_scaled):raise AssertionError('f_C mismatch')
        check(name+' mechanism disjoint unions and matched allocation',True)
        check(name+' independent FPC predictions',True)
        check(name+' signed mass identity',close(dc.candidate_scaled-dc.truncated_scaled,dc.extra_scaled-dc.missing_scaled))
        check(name+' full error identity',close(draws.total_error_scaled-draws.sampling_error_scaled,(draws.query_id.map(dc.set_index('query_id').candidate_scaled)-draws.query_id.map(dc.set_index('query_id').full_scaled))))
        details[name]['mechanism']={'draws':len(draws),'max_abs_mean_z':float(sm.mean_z.abs().max()),'mean_z_abs_gt3':int((sm.mean_z.abs()>3).sum()),'nonzero_prediction_cases':int((sm.predicted_variance_scaled>0).sum()),'empirical_predicted_ratio_median':float(sm.variance_ratio.median()),'empirical_predicted_ratio_p05':float(sm.variance_ratio.quantile(.05)),'empirical_predicted_ratio_p95':float(sm.variance_ratio.quantile(.95))}
    print(json.dumps({'audited':name,'details':details[name]}),flush=True)
report={'status':'PASS','checks':checks,'details':details,'code_commit':r.git_head(),'run_manifest_sha256':r.sha(b/'run_manifest.json'),'note':'Mean z and variance-ratio distributions are descriptive Monte Carlo diagnostics, not simultaneous confidence guarantees. Timing equality checks medians of saved repeats, not new timings.'}
(out/'audit_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'status':'PASS','checks':len(checks)}))
