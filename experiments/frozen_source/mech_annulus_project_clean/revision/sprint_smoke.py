import json,time
import numpy as np,pandas as pd
import core as c
from sprint_common import load_data,ROOT,DATASETS,save_json
rows=[]
for name in DATASETS:
 d=load_data(name);e=d['old']
 saved=pd.read_csv(ROOT/'prior_final/studies/c5_L12_b2_D12_J8_epoch24'/name/'queries.csv')
 for qi,q in enumerate(d['arrays']['validation'][:5]):
  truth=c.exact_log_mean(e.points,q,e.h)
  for seed in [0,1]:
   z=e.evaluate('approx_annular_mech',q,128,np.random.default_rng(seed+qi*100003))
   old=saved[(saved.query_id==qi)&(saved.seed==seed)&(saved.budget==128)].iloc[0]
   rows.append(dict(dataset=name,query_id=qi,raw_row_id=int(d['split']['validation'][qi]),seed=seed,log_estimate=z['log_estimate'],saved_log_estimate=old.log_estimate,estimate_pass=bool(np.isclose(z['log_estimate'],old.log_estimate,rtol=0,atol=1e-12)),truth_pass=bool(np.isclose(truth,old.log_truth,rtol=0,atol=1e-12)),budget_pass=z['actual_samples']==128,time_ms=z['time_ns']/1e6))
 print(name,'smoke complete',flush=True)
out=ROOT/'results/smoke';out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/'baseline_smoke.csv',index=False)
assert all(x['estimate_pass'] and x['truth_pass'] and x['budget_pass'] for x in rows)
save_json(out/'PASS.json',dict(checks=3*len(rows),queries=20,all_passed=True))
