"""Independent local verification from saved values, membership IDs and sampling outputs."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
DS=['isolet','cifar10','cifar10_gist512','amazon']
checks={};maxdiff=0.;ndraw=0
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=json.loads((R/'BASELINE_FREEZE.json').read_text())
assert all(sha(p)==h for p,h in freeze['files'].items())
checks['frozen_manuscript_files_unchanged']=len(freeze['files'])
protocol=json.loads((R/'PROTOCOL_FREEZE.json').read_text())
assert all(sha(R/p)==h for p,h in protocol['sha256'].items())
checks['predeclared_protocol_unchanged']=True
for stage in ['fixed_union','degradation']:
 start=json.loads((R/'results'/stage/'START.json').read_text())
 assert start['protocol_sha256']==sha(R/'EXPERIMENT_ROADMAP.md')
 assert start['script_sha256']==sha(R/'scripts/mechanism.py')
 checks[stage+'_source_and_protocol_match']=True
 count=0
 for ds in DS:
  folder=R/'results'/stage
  z=pd.read_csv(folder/f'{ds}_query_metrics.csv');draws=np.load(folder/f'{ds}_estimates.npz')
  cache=np.load(R/'results/fixed_union'/f'{ds}_cache.npz')
  degradation=np.load(folder/f'{ds}_partitions.npz') if stage=='degradation' else None
  assert len(z)==(80*26 if stage=='fixed_union' else 80*25)
  assert set(z.query_id)==set(range(80))
  assert (z.actual_samples==128).all() and (z.repetitions==1024).all()
  assert z.groupby(['query_id','block']).candidate_hash.nunique().max()==1
  assert z.groupby(['query_id','block']).H_hash.nunique().max()==1
  assert z.groupby(['query_id','block']).candidate_kernel_mass_recall.nunique().max()==1
  multi=z[z.partition!='single']
  assert multi.groupby(['query_id','block']).group_sizes.nunique().max()==1
  assert multi.groupby(['query_id','block']).allocation.nunique().max()==1
  for t in z.itertuples():
   qi=t.query_id;values=cache[f'q{qi}_values'];H=cache[f'q{qi}_H'] if t.block=='matched_H' else np.array([],dtype=int)
   C=np.arange(len(values)) if t.block=='matched_H' else cache[f'q{qi}_C'];total=values[C].sum()
   if stage=='fixed_union':
    key=f'{qi}_{t.block}_{t.partition}_{t.partition_repeat}'
    order=np.setdiff1d(C,H) if t.partition=='single' else cache[key+'_partition']
   else:
    key=f'{qi}_{t.partition_repeat}_{t.level:g}';order=degradation[key]
   sizes=np.array(json.loads(t.group_sizes));counts=np.array(json.loads(t.allocation))
   assert np.array_equal(np.sort(np.concatenate([order,H])),np.sort(C))
   assert len(np.unique(np.concatenate([order,H])))==len(C)
   groups=np.split(order,np.cumsum(sizes)[:-1]);pvar=0.
   for ids,n,m in zip(groups,sizes,counts):
    assert len(ids)==n and 0<m<=n
    yy=values[ids]
    ss=sum((float(y)-float(yy.mean()))**2 for y in yy)/(n-1) if n>1 else 0.
    pvar+=(n*n/m)*(1-m/n)*ss/total**2
   est=draws[key];assert len(est)==1024 and np.isfinite(est).all()
   diff=abs(pvar-t.predicted_relative_variance);maxdiff=max(maxdiff,diff)
   assert math.isclose(pvar,t.predicted_relative_variance,rel_tol=1e-9,abs_tol=1e-13)
   assert math.isclose(est.var(ddof=1),t.empirical_relative_variance,rel_tol=1e-9,abs_tol=1e-13)
   assert math.isclose(np.abs(est-1).mean(),t.mean_relative_error,rel_tol=1e-9,abs_tol=1e-13)
   ndraw+=len(est);count+=1
 checks[stage+'_verified_query_partition_rows']=count
checks['independently_recomputed_variance_max_abs_difference']=maxdiff
checks['saved_estimator_realizations']=ndraw
qa=json.loads((R/'results/fixed_union/CORRECTNESS.json').read_text())
assert len(qa)==24 and max(x['abs_diff'] for x in qa)<1e-10
checks['unchanged_production_estimator_crosschecks']=len(qa)
checks['production_formula_max_abs_difference']=max(x['abs_diff'] for x in qa)
(R/'qa/VERIFICATION.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
print(json.dumps(checks,indent=2))
