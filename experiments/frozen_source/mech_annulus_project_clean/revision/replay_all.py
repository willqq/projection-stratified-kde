"""Run the frozen grid in a NEW remote replay root, never over saved results.
Create it with git clone of source_history.bundle, copy run_manifest*.json,
install the environment/build DEANN, and provide worktree/data as documented.
"""
from pathlib import Path
import os,subprocess,sys,json,time
b=Path(__file__).resolve().parents[3]
p=b/'worktree/mech_annulus_project_clean'
manifest=json.loads((b/'run_manifest.json').read_text())
if (b/'results').exists() or (b/'assets').exists():raise SystemExit('Use a fresh replay root without results or assets; original records will not be overwritten.')
env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(b/'.venv/lib'))
logdir=b/'replay_logs';logdir.mkdir(exist_ok=False)
jobs=[['-m','pytest','-q','revision/test_core.py','revision/test_deann.py']]
jobs += [['revision/run.py','--phase','pilot','--dataset','isolet','--run-id','pilot_isolet_v1']]
jobs += [['revision/run.py','--phase','formal','--dataset',name,'--run-id','basic_v1'] for name in manifest['datasets']]
jobs += [['revision/mechanism.py','--dataset',name,'--run-id','mechanism_v1'] for name in manifest['mechanism']['datasets']]
jobs += [['revision/run_deann.py','--dataset',name,'--run-id','deann_v1'] for name in manifest['datasets']]
jobs += [['revision/audit_results.py'],['revision/aggregate_plot.py']]
status=[]
for i,args in enumerate(jobs):
    cmd=[sys.executable,'-u']+args
    with (logdir/f'{i:02d}.log').open('w') as log:
        ret=subprocess.run(cmd,cwd=p,env=env,stdout=log,stderr=log)
    status.append({'command':cmd,'exit_code':ret.returncode,'completed':time.time()})
    (logdir/'status.json').write_text(json.dumps(status,indent=2))
    if ret.returncode:raise SystemExit(ret.returncode)
print('Frozen replay completed; new timing values are machine- and run-specific.')
