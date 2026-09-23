"""Independent artifact checks; this script performs no estimation or selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
checks={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(2**22),b''):h.update(x)
 return h.hexdigest()

manifest=json.loads((ROOT/'baseline/MANIFEST.json').read_text())
changed=[x['path'] for x in manifest['files'] if sha(Path(x['path']))!=x['sha256']]
assert not changed,changed
checks['baseline_unchanged_files']=len(manifest['files'])
assert sha(ROOT/'EXPERIMENT_DECISION.md')==manifest['protocol_sha256']
checks['protocol_unchanged']=True
for item in json.loads((R/'REMOTE_MANIFEST.json').read_text()):
 assert sha(R/item['path'])==item['sha256'],item['path']
checks['remote_artifacts_match_hashes']=175
for stage in ['bandwidth','deann_validation','deann_formal','scaling']:
 start=json.loads((R/(stage+'_START.json')).read_text())
 source=R/('enhance_scaling.py' if stage=='scaling' else 'enhance_bandwidth_fairness.py')
 assert start['script_sha']==sha(source)
checks['executed_script_versions_match']=True

band_rows=0
for f in (R/'bandwidth').glob('*_queries.csv'):
 z=pd.read_csv(f);band_rows+=len(z)
 assert len(z)==20000
 assert z[['query_id','h_factor','method','seed']].duplicated().sum()==0
 assert set(z.h_factor)=={.5,.75,1,1.5,2} and set(z.seed)==set(range(5))
 assert z.groupby('query_id').H_hash.nunique().max()==1
 assert (z.actual_samples==128).all() and (z.kernel_evaluations==128).all()
 assert np.isfinite(z.relative_error).all() and (z.truth_float_underflow==0).all()
assert band_rows==140000
checks['bandwidth_rows']=band_rows

fr=json.loads((R/'DEANN_FROZEN.json').read_text())
start=json.loads((R/'deann_formal_START.json').read_text())
assert fr['created_unix']<start['unix']
checks['deann_frozen_before_formal']=True
for ds,v in fr['datasets'].items():
 grid=pd.read_csv(R/'deann_validation'/ds/'grid.csv')
 assert len(grid)==24
 expected=[];seen=set()
 for metric in ['mean_relative_error','median_time_ns','error_time_product']:
  z=grid.sort_values(list(dict.fromkeys([metric,'median_time_ns','nprobe','near_fraction']))).iloc[0]
  pair=(int(z.nprobe),float(z.near_fraction))
  if pair not in seen:expected.append(pair);seen.add(pair)
 assert expected==[(v['nprobe'],v['near_fraction']) for v in v['anchors']]
checks['validation_selection_independently_reconstructed']=True

formal_counts={}
for stage in ['deann_formal','scaling']:
 total=0
 for done in (R/stage).rglob('DONE.json'):
  d=json.loads(done.read_text());folder=done.parent;q=d['queries'];budgets=d['budgets']
  basic=pd.read_csv(folder/'basic_queries.csv');ann=pd.read_csv(folder/'deann_queries.csv')
  assert len(basic)==q*(1+3*len(budgets)*5)
  assert len(ann)==q*len(budgets)*5*len(d['variants'])
  for z in [basic,ann]:
   assert not z[['method','budget','seed','query_id']].duplicated().any()
   assert np.isfinite(z.relative_error).all()
  final=basic[basic.method=='final']
  assert (final.actual_samples==final.budget).all() and (final.kernel_evaluations==final.budget).all()
  assert (final.H_count==final.budget//4).all()
  assert (ann.kernel_evaluations>=ann.actual_samples).all()
  t=pd.read_csv(folder/'basic_timings.csv');t=t[t.method=='final']
  components=['projection_ns','low_dim_score_ns','sorting_and_strata_ns','residual_build_ns','sampling_ns','allocation_ns','kernel_ns','aggregation_ns','unassigned_ns']
  assert np.allclose(t[components].sum(axis=1),t.time_ns)
  assert d['basic_replay_max']<1e-11 and d['deann_replay_max']<1e-9
  total+=len(basic)+len(ann)
 formal_counts[stage]=total
assert len(list((R/'deann_formal').rglob('DONE.json')))==7
assert len(list((R/'scaling').rglob('DONE.json')))==5
checks['formal_row_counts']=formal_counts

ids=json.loads((R/'scaling_ids.json').read_text())
assert len(ids['reference_order'])==len(set(ids['reference_order']))==20000
assert len(ids['query_ids'])==100 and not set(ids['reference_order'])&set(ids['query_ids'])
splits=json.loads((R/'inputs/cifar10_gist512/split_ids.json').read_text())
locked=json.loads((R/'inputs/audit_locked_ids.json').read_text())['datasets']['cifar10_gist512']['audit_ids']
assert ids['query_ids']==locked[:100] and ids['reference_order'][:1800]==splits['reference']
excluded=set(locked)|set(splits['train'])|set(splits['validation'])|set(splits['test'])
assert not excluded&set(ids['reference_order'])
for n in [1000,1800,5000,10000,20000]:
 tr=pd.read_csv(R/'scaling'/str(n)/'truth.csv')
 assert tr.raw_row_id.tolist()==ids['query_ids']
 sel=json.loads((R/'scaling_validation'/str(n)/'selection.json').read_text())
 assert sel['created_unix']<json.loads((R/'scaling'/str(n)/'START.json').read_text())['created_unix']
checks['scaling_query_set_and_nested_reference_ids']=True
old=pd.read_csv(ROOT.parent/'sprint_final/results/formal/audit/cifar10_gist512/basic_queries.csv')
new=pd.read_csv(R/'scaling/1800/basic_queries.csv')
joined=new.merge(old,on=['method','query_id','budget','seed'],suffixes=('_new','_old'))
diff=np.max(np.abs(joined.relative_error_new-joined.relative_error_old))
assert diff<1e-11
checks['scaling_n1800_replay_max_error_difference']=float(diff)

(ROOT/'qa/VERIFICATION.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
print(json.dumps(checks,indent=2))
