"""Write the immutable method and provenance before any formal query evaluation."""
import json,time,subprocess
from pathlib import Path
import run as r
from sprint_common import ROOT,DATASETS,save_json
assert not (ROOT/'freeze_manifest.json').exists()
choice=json.loads((ROOT/'results/validation/selection_decision.json').read_text())
old=json.loads((ROOT/'prior_final/final_config.yaml').read_text())
source=Path(__file__).parent
hashes={p.name:r.sha(p) for p in sorted(source.iterdir()) if p.suffix in ('.py','.cpp','.so')}
assets={str(p.relative_to(ROOT)):r.sha(p) for p in sorted((ROOT/'models').rglob('*')) if p.is_file()}
f=dict(frozen_before_formal_evaluation=True,freeze_time_unix=time.time(),source_commit=r.git_head(),source_hashes=hashes,method_config=choice['method_config'],selected=choice['selected'],
 validation_selection_sha256=r.sha(ROOT/'results/validation/selection_decision.json'),audit_ids_sha256=r.sha(ROOT/'audit/locked_ids.json'),model_and_calibration_hashes=assets,
 projection=dict(dimension=64,seed=20260916,distribution='Gaussian N(0,1/64)',training_mean_centering=True,reference_scan='all reference projections',kernel_space='unchanged original transformed d-dimensional features'),
 budgets=[32,64,128,256,512],sampling_seeds=list(range(5)),test_queries=200,audit_queries={'isolet':200,'cifar10':200,'cifar10_gist512':200,'amazon':0},timing_repeats=3,threads=8,
 old_test_status='Previously exposed, used only for controlled new/old comparison; audit locked before sprint validation',
 timing_boundary='Same transformed query vector for every method. All online projection/retrieval/sorting/strata/allocation/kernel steps included. Raw transform measured separately. No concurrent experimental computation.',
 basic_rng_rule='seed+query_id*100003; reset for each of three timing repeats',deann_rng_rule='Official permuted estimator; one warmup, three advancing sampler calls per query; first value supplies error',
 matched_control='Same H and total M. Projection single remainder reuses ranked[k:]. Final proportional rows reused as identical matched_proportional control; Neyman separately measured.',
 datasets=old['datasets'])
for name in DATASETS:
    meta=json.loads((r.ASSETS/name/'metadata.json').read_text())
    for key in ['data_sha256','split_sha256','preprocessing_sha256']:assert meta[key]==f['datasets'][name][key]
    f['datasets'][name]['legacy_c5_model_sha256']=r.sha(r.ASSETS/name/'model.pt')
    f['datasets'][name]['legacy_source_model_note']='For current_approx comparator only; final estimator uses projection64 cache'
    f['datasets'][name].pop('model_relative_path',None)
save_json(ROOT/'final_config.yaml',f)
save_json(ROOT/'freeze_manifest.json',dict(frozen=True,created_unix=time.time(),source_commit=r.git_head(),final_config_sha256=r.sha(ROOT/'final_config.yaml'),method_config=choice['method_config']))
print(json.dumps(dict(frozen=True,source_commit=r.git_head(),method_config=f['method_config'],time_unix=f['freeze_time_unix'])))
