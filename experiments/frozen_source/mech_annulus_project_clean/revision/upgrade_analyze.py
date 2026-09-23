"""Query-cluster inference and prespecified decision tables; no tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def paired_gain(a,b,kind='mean',resamples=1000):
    """Positive means b improves on a; query rows are the resampling unit."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    assert a.shape==b.shape and len(a)>1
    rng=np.random.default_rng(20260920)
    ix=rng.integers(0,len(a),size=(resamples,len(a)))
    agg=np.mean if kind=='mean' else np.median
    value=1-agg(b)/agg(a)
    axes=tuple(range(1,a.ndim+1))
    distribution=1-agg(b[ix],axis=axes)/agg(a[ix],axis=axes)
    lo,hi=np.quantile(distribution,[.025,.975])
    return dict(relative_reduction=float(value),ci_low=float(lo),ci_high=float(hi),queries=len(a))


def adjacent(budgets):
    budgets=set(budgets)
    return any(a in budgets and b in budgets for a,b in [(32,64),(64,128),(128,256),(256,512)])


def analyze(inputs,out):
    out.mkdir(parents=True,exist_ok=False)
    summaries=[];ann_gains=[];speed_gains=[];variance=[];profiles=[]
    for root in inputs:
        if not (root/'DONE.json').exists():raise RuntimeError('Incomplete run: '+str(root))
        done=json.loads((root/'DONE.json').read_text())
        if done.get('scope')=='smoke only':raise RuntimeError('Smoke cannot make a scientific decision')
        for folder in root.iterdir():
            if not folder.is_dir() or not (folder/'raw_queries.jsonl').exists():continue
            offline=json.loads((folder/'OFFLINE.json').read_text())
            df=pd.read_json(folder/'raw_queries.jsonl',lines=True)
            if set(df.seed.unique())!={0,1,2,3,4}:raise RuntimeError('Missing sampling seeds')
            if set(df.repeat.unique())!={0,1,2}:raise RuntimeError('Missing timing repetitions')
            prefix=dict(dataset=folder.name,phase=offline['phase'],n=offline['n'])
            ids=['method','budget','query_id','seed']
            # Repetition zero supplies estimates; all three repetitions supply
            # median latency. The official stateful sampler may change by rep.
            stats=df[df.repeat==0].set_index(ids)
            stats['median_time_ns']=df.groupby(ids).time_ns.median()
            stats=stats.reset_index()
            for (method,M),z in stats.groupby(['method','budget']):
                q=z.groupby('query_id').relative_error.mean()
                summaries.append(dict(**prefix,method=method,budget=M,
                    mean_relative_error=q.mean(),median_error=q.median(),p90_error=q.quantile(.9),
                    se_query_cluster=q.std(ddof=1)/np.sqrt(len(q)),
                    median_latency_ms=z.median_time_ns.median()/1e6,queries=len(q),
                    actual_kernel_evaluations=z.kernel_evaluations.mean(),
                    actual_samples=z.actual_samples.mean()))
            for M,z in stats.groupby('budget'):
                eq=z.groupby(['query_id','method']).relative_error.mean().unstack()
                tq=z.pivot(index=['query_id','seed'],columns='method',values='median_time_ns').sort_index()
                if 'projected_fast' in tq:
                    speed_gains.append(dict(**prefix,budget=M,
                        **paired_gain(tq.projected_current.to_numpy().reshape(-1,5),
                                      tq.projected_fast.to_numpy().reshape(-1,5),kind='median')))
                for column in eq:
                    if column.startswith('ann_') and column.endswith('_single'):
                        other=column.replace('_single','_multi')
                        if other in eq:
                            ann_gains.append(dict(**prefix,budget=M,anchor=column.split('_')[1],
                                single_error=eq[column].mean(),multi_error=eq[other].mean(),
                                **paired_gain(eq[column],eq[other])))
            file=folder/'ann_variance.jsonl'
            if file.stat().st_size:
                vd=pd.read_json(file,lines=True)
                for (anchor,M),z in vd.groupby(['anchor','budget']):
                    variance.append(dict(**prefix,anchor=anchor,budget=M,
                        median_variance_ratio=z.variance_ratio.median(),
                        mean_variance_ratio=z.variance_ratio.mean(),
                        queries=len(z)))
            pdg=pd.read_json(folder/'current_profile.jsonl',lines=True)
            profiles.append(dict(**prefix,**{k:pdg[k].mean()/1e6 for k in pdg if k.endswith('_ns')}))
    S=pd.DataFrame(summaries);A=pd.DataFrame(speed_gains);B=pd.DataFrame(ann_gains);V=pd.DataFrame(variance)
    S.to_csv(out/'summary.csv',index=False);A.to_csv(out/'path_a_timing_bootstrap.csv',index=False)
    B.to_csv(out/'ann_H_paired_bootstrap.csv',index=False);V.to_csv(out/'ann_H_variance.csv',index=False)
    pd.DataFrame(profiles).rename(columns=lambda s:s.replace('_ns','_mean_ms')).to_csv(out/'current_profile_mean_ms.csv',index=False)
    targets=[];caps=[]
    for keys,z in S.groupby(['dataset','phase','n']):
        prefix=dict(zip(['dataset','phase','n'],keys))
        for method,curve in z.groupby('method'):
            if method=='exact':
                # One exact algorithm; avoid choosing its fastest repeated M.
                curve=pd.DataFrame([dict(method=method,budget=0,mean_relative_error=0.,
                    median_latency_ms=curve.median_latency_ms.median())])
            for target in [.05,.10,.15,.20]:
                eligible=curve[curve.mean_relative_error<=target]
                chosen=eligible.sort_values('median_latency_ms').iloc[0] if len(eligible) else None
                targets.append(dict(**prefix,method=method,error_target=target,
                    attained=chosen is not None,budget=int(chosen.budget) if chosen is not None else None,
                    latency_ms=float(chosen.median_latency_ms) if chosen is not None else None))
            for cap in [.125,.25,.5,1.,2.,4.]:
                eligible=curve[curve.median_latency_ms<=cap]
                chosen=eligible.sort_values('mean_relative_error').iloc[0] if len(eligible) else None
                caps.append(dict(**prefix,method=method,time_cap_ms=cap,
                    attained=chosen is not None,budget=int(chosen.budget) if chosen is not None else None,
                    error=float(chosen.mean_relative_error) if chosen is not None else None))
    pd.DataFrame(targets).to_csv(out/'fixed_error_latency.csv',index=False)
    pd.DataFrame(caps).to_csv(out/'fixed_time_error.csv',index=False)
    gate=dict(path_a=False,path_b=False,scope='validation feasibility only',qualified_a=[],qualified_b=[])
    if len(A):
        for keys,z in A[A.phase=='validation'].groupby(['dataset','n']):
            good=z[z.relative_reduction>=.20]
            if adjacent(good.budget):
                gate['path_a']=True;gate['qualified_a'].append(dict(dataset=keys[0],n=int(keys[1]),budgets=good.budget.tolist()))
    if len(B):
        joined=B.merge(V,on=['dataset','phase','n','anchor','budget'],validate='one_to_one',suffixes=('','_variance'))
        for keys,z in joined[joined.phase=='validation'].groupby(['dataset','n','anchor']):
            good=z[(z.relative_reduction>=.10)&(z.ci_low>0)&(z.median_variance_ratio<=.90)]
            if adjacent(good.budget):
                gate['path_b']=True;gate['qualified_b'].append(dict(dataset=keys[0],n=int(keys[1]),anchor=keys[2],budgets=good.budget.tolist()))
    if not (S.phase=='validation').any():
        gate=dict(path_a=None,path_b=None,scope='Not applicable: inputs contain formal sets only; use MERGE_GATE.json',
                  qualified_a=[],qualified_b=[])
    (out/'VALIDATION_GATE.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps(gate,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();analyze(args.input,args.out)
