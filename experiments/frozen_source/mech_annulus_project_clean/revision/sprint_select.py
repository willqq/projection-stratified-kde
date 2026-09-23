"""Validation-only final selection and matched-component effect record."""
import json,time
import numpy as np,pandas as pd
from sprint_common import ROOT,DATASETS,save_json
V=ROOT/'results/validation';s=pd.read_csv(V/'finalist_summary.csv');configs=json.loads((V/'screen_configs.json').read_text());rows=[]
for key,g in s.groupby('config'):
    w=g.pivot(index='dataset',columns='budget',values='recovery')
    qualified=((w[64]>=.5)&(w[128]>=.5))|((w[128]>=.5)&(w[256]>=.5))
    rows.append(dict(config=key,qualifying_datasets=int(qualified.sum()),median_recovery=float(g.recovery.median()),geomean_latency_ms=float(np.exp(np.log(g.median_time_ms).mean()))))
compare=[]
for key,parent in [('b1a_projection64_neyman','b1a_projection64'),('b1a_projection64','matched_projection64')]:
    a=s[s.config==key].set_index(['dataset','budget']);b=s[s.config==parent].set_index(['dataset','budget'])
    # All validation query/seed/budget observations have equal weight in error means.
    bydataset=(a.mean_relative_error.groupby('dataset').mean()/b.mean_relative_error.groupby('dataset').mean())
    ratio=a.mean_relative_error.mean()/b.mean_relative_error.mean()
    passed=bool(ratio<.98 and (bydataset<1).sum()>=2 and (bydataset<=1.05).all())
    compare.append(dict(config=key,parent=parent,pooled_error_ratio=float(ratio),per_dataset_error_ratios=bydataset.to_dict(),increment_constraint_passed=passed))
assert compare[0]['increment_constraint_passed']==False
assert compare[1]['increment_constraint_passed']==True
chosen='b1a_projection64'
boot=[];rng=np.random.default_rng(20260916)
for dataset in DATASETS:
    frames={k:pd.read_csv(V/dataset/(k+'.csv')) for k in ['uniform_mc','exact_annular','b1a_projection64','matched_projection64','b1a_projection64_neyman']}
    for M in [64,128,256]:
        means={k:f[f.budget==M].groupby('query_id').relative_error.mean().sort_index().to_numpy() for k,f in frames.items()}
        ids=rng.integers(0,80,size=(1000,80))
        for other in ['matched_projection64','b1a_projection64_neyman']:
            dif=means[chosen]-means[other];samples=dif[ids].mean(axis=1)
            boot.append(dict(dataset=dataset,budget=M,contrast=chosen+' minus '+other,mean_error_difference=float(dif.mean()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),resampling_unit='query; five seeds averaged within query',resamples=1000))
record=dict(selected=chosen,method_config=configs[chosen],candidate_method_count=11,mandatory_controls=['matched_Hquarter','matched_projection64'],selection_time_unix=time.time(),formal_queries_evaluated=False,candidate_summary=rows,component_constraints=compare,
 reason='Both projection candidates qualify on all four datasets. Neyman has higher median Recovery but fails the per-dataset error guard on Amazon when pooling all validation budgets equally. Proportional projection strata pass the independent matched-H gain requirement on all four datasets and retain the simpler lower-latency estimator.',
 clarification='Per-dataset component guard uses mean error over all query/seed/budget observations (ratio of dataset means), not an average of per-budget ratios. No test/audit selection.')
save_json(V/'selection_decision.json',record);pd.DataFrame(boot).to_csv(V/'matched_bootstrap.csv',index=False);print(json.dumps(record,indent=2))
