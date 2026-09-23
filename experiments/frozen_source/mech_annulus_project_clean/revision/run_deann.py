from __future__ import annotations
import argparse
import json
import time
import numpy as np
import pandas as pd
import core as c
import run as r
import deann_adapter as d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--run-id',required=True)
    args=ap.parse_args()
    manifest=json.loads((r.JOBROOT/'run_manifest.json').read_text())
    assert manifest['frozen'] and args.dataset in manifest['datasets']
    cfg=manifest['deann']
    arrays,split,meta,e,_=r.setup(args.dataset)
    out=r.JOBROOT/'results'/args.run_id
    out.mkdir(parents=True,exist_ok=True)
    dest=out/(args.dataset+'_deann_queries.csv')
    if dest.exists():raise RuntimeError('Do not overwrite a DEANN run')
    ann,offline=d.build(e.points)
    # Fixed small validation grid, fully specified before test results.
    val=[]
    for probe in cfg['nprobe_validation_candidates']:
        for fraction in cfg['near_fraction_validation_candidates']:
            est,k,m,fit=d.make_estimator(e.points,e.h,ann,cfg['validation_budget'],fraction,probe,0)
            for qi,q in enumerate(arrays['validation'][:cfg['validation_queries']]):
                result=d.evaluate(est,ann,q,k,m)
                result.update(c.errors(result.pop('log_estimate'),c.exact_log_mean(e.points,q,e.h)))
                val.append(dict(nprobe=probe,near_fraction=fraction,query_id=qi,**result))
    vf=pd.DataFrame(val)
    score=vf.groupby(['nprobe','near_fraction'],as_index=False).agg(relative_error=('relative_error','mean'),time_ns=('time_ns','mean'))
    chosen=score.sort_values(['relative_error','time_ns','nprobe','near_fraction'],kind='stable').iloc[0]
    probe,fraction=int(chosen.nprobe),float(chosen.near_fraction)
    vf.to_csv(out/(args.dataset+'_deann_validation.csv'),index=False)
    score.to_csv(out/(args.dataset+'_deann_validation_summary.csv'),index=False)
    selected={'nprobe':probe,'near_fraction':fraction,'nlist':offline['nlist'],
              'selected_before_test':True,'source':cfg,'code_commit':r.git_head(),
              'shared_reference_sha256':meta['data_sha256'],'model_seed':r.MODEL_SEED,
              'deann_extension_sha256':r.sha(d.libs[0])}
    (out/(args.dataset+'_deann_config.json')).write_text(json.dumps(selected,indent=2))
    print(json.dumps({'event':'deann_validation_frozen','dataset':args.dataset,'nprobe':probe,'near_fraction':fraction}),flush=True)
    rows=[];timings=[];fit_rows=[]
    # Truth belongs to the evaluator; never passed into official DEANN/ANN.
    queries=arrays['test'][:manifest['test_queries']]
    truths=[c.exact_log_mean(e.points,q,e.h) for q in queries]
    for budget in manifest['budgets']:
        for seed in manifest['sampling_seeds']:
            est,k,m,fit=d.make_estimator(e.points,e.h,ann,budget,fraction,probe,seed)
            fit_rows.append(dict(budget=budget,seed=seed,permutation_fit_seconds=fit))
            d.evaluate(est,ann,queries[0],k,m)  # one-query warmup for this instance
            for qi,q in enumerate(queries):
                reps=[]
                for rep in range(manifest['timing_repeats']):
                    result=d.evaluate(est,ann,q,k,m)
                    reps.append(result)
                    timings.append(dict(dataset=args.dataset,method='deann',query_id=qi,
                        raw_row_id=int(split['test'][qi]),budget=budget,seed=seed,repeat=rep,**result))
                result=reps[0].copy()
                for key in ['time_ns','candidate_ns','encoding_ns','radius_ns','allocation_ns','kernel_ns']:
                    result[key]=float(np.median([x[key] for x in reps]))
                result.update(c.errors(result.pop('log_estimate'),truths[qi]))
                rows.append(dict(dataset=args.dataset,method='deann',query_id=qi,
                    raw_row_id=int(split['test'][qi]),budget=budget,seed=seed,**result))
        print(json.dumps({'event':'deann_budget_complete','dataset':args.dataset,'budget':budget}),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(dest,index=False)
    pd.DataFrame(timings).to_csv(out/(args.dataset+'_deann_timings.csv'),index=False)
    c.aggregate(frame).to_csv(out/(args.dataset+'_deann_summary.csv'),index=False)
    pd.DataFrame(fit_rows).to_csv(out/(args.dataset+'_deann_offline_fits.csv'),index=False)
    (out/(args.dataset+'_deann_offline.json')).write_text(json.dumps(offline,indent=2))
    (out/(args.dataset+'_deann_DONE.json')).write_text(json.dumps({'rows':len(rows),'code_commit':r.git_head(),'completed':time.time()}))

if __name__=='__main__':main()
