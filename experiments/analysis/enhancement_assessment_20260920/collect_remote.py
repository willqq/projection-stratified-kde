from pathlib import Path
import hashlib,json,platform,shutil,subprocess,time
import numpy,scipy,torch,faiss
root=Path('/root/autodl-tmp/icassp_enhancement_20260920')
out=root/'enhancement';inputs=out/'inputs';inputs.mkdir(exist_ok=True)
for name in ['isolet','cifar10','cifar10_gist512','amazon']:
    d=inputs/name;d.mkdir(exist_ok=True)
    for file in ['metadata.json','split_ids.json']:shutil.copy2(root/'assets'/name/file,d/file)
for src,name in [(root/'audit/locked_ids.json','audit_locked_ids.json'),(root/'final_config.yaml','original_final_config.json'),(root/'freeze_manifest.json','original_freeze_manifest.json')]:shutil.copy2(src,inputs/name)
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True)
env=dict(created_unix=time.time(),python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,torch=torch.__version__,faiss=faiss.__version__,
    cpu=subprocess.check_output(['lscpu'],text=True),execution_threads=8,cuda_visible_devices='',
    baseline_states={str(p):dict(head=git(p,'rev-parse','HEAD').strip(),status=git(p,'status','--short')) for p in [Path('/root/autodl-tmp/icassp_sprint_20260916_0103/worktree'),Path('/root/autodl-tmp/icassp_structural_20260919/worktree'),root/'worktree']})
(out/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
(out/'enhancement.patch').write_text(git(root/'worktree','diff','6741795be077b813cedd5ed888a5a692fb266b6e','HEAD'))
(out/'COMMITS.txt').write_text(git(root/'worktree','log','-3','--format=fuller'))
(out/'enhance_bandwidth_fairness.py').write_text(git(root/'worktree','show','5e5b492:mech_annulus_project_clean/revision/enhance.py'))
shutil.copy2(root/'worktree/mech_annulus_project_clean/revision/enhance.py',out/'enhance_scaling.py')
checks=[]
for p in sorted(out.rglob('*')):
 if p.is_file() and p.name!='REMOTE_MANIFEST.json':
  h=hashlib.sha256(p.read_bytes()).hexdigest();checks.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=h))
(out/'REMOTE_MANIFEST.json').write_text(json.dumps(checks,indent=2))
print(len(checks),'files hashed')
