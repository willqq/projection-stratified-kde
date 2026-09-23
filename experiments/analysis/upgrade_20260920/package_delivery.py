from pathlib import Path
import json,hashlib,zipfile,tarfile,subprocess
from datetime import datetime,timezone
import pdfplumber
R=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
expected={'remote_source.tar.gz':'4f1a3c87897825ff8cf6df2040809ff1e4e731fbef4247f186ec551866591714','remote_upgrade.patch':'881b4ace79279c8e02fa7a4e311ea15f9b31c4c9069f1de7cd83a75f047f45a3'}
assert all(sha(R/f)==h for f,h in expected.items())
oldlines=(R/'DELIVERY_SHA256.txt').read_text().splitlines(); oldraw=next(line.split()[0] for line in oldlines if 'raw_remote_results.tar.gz' in line)
assert sha(R/'raw_remote_results.tar.gz')==oldraw
expected['raw_remote_results.tar.gz']=oldraw
(R/'DELIVERY_SHA256.txt').write_text(''.join(h+'  '+f+'\n' for f,h in expected.items()))
with tarfile.open(R/'remote_source.tar.gz') as t:
 for p in (R/'worktree/source').glob('upgrade_*.py'):
  remote='mech_annulus_project_clean/revision/'+p.name
  assert t.extractfile(remote).read()==p.read_bytes(),p.name
state={'formal_experiment_commit':'f872802ae4cca95ec69376d548c8389627de8761','remote_delivery_commit':'757d1801748d5b19ca4940d7a99d89d0b9f86ed8','local_snapshot_commit':subprocess.check_output(['git','-C',str(R/'worktree'),'rev-parse','HEAD'],text=True).strip(),'raw_archive_reporting_snapshot':'94d6bb8','post_formal_changes':'Analysis, checks and figures only; raw experiment source and results unchanged.','source_identity':'All upgrade_*.py match local and delivered remote source bytes.','sha256':expected}
(R/'SOURCE_STATE.json').write_text(json.dumps(state,indent=2))
figs=[]
for p in sorted((R/'figures').glob('*.pdf')):
 with pdfplumber.open(p) as pdf:
  for page in pdf.pages:
   chars=[c for c in page.chars if c['text'].strip()]
   minsize=min(c['size'] if c['upright'] else c['width'] for c in chars)
   inside=all(c['x0']>=-.01 and c['x1']<=page.width+.01 and c['top']>=-.01 and c['bottom']<=page.height+.01 for c in chars)
   assert minsize>=8.999 and inside,(p,minsize,inside)
   figs.append({'file':str(p.relative_to(R)),'min_font_pdf_pt':minsize,'text_within_page':inside})
(R/'qa/FIGURE_CHECKS.json').write_text(json.dumps(figs,indent=2))
roots=['manuscript','scripts','protocol','results','figures','remote_metadata','worktree/source','baseline_backup/final_verification_20260920/manuscript']
files=set()
for name in roots:
 files.update(p for p in (R/name).rglob('*') if p.is_file() and p.name!='.DS_Store' and '__pycache__' not in p.parts)
files.update(p for p in R.glob('*.md'))
files.update(R/f for f in ['SOURCE_STATE.json','UPGRADE_FREEZE.json','MANUSCRIPT.diff','MANUSCRIPT_CHANGE_RECORD.json','BACKUP_MANIFEST.json','DELIVERY_SHA256.txt','remote_upgrade.patch','upgrade.patch','remote_source.tar.gz'])
files.update(R/'qa'/f for f in ['DELIVERY_CHECKS.json','FIGURE_CHECKS.json','ARCHIVE_VERIFIED.json','fonts.txt','compile.log'])
# Root delivery package is portable; raw query records remain a separate verified archive.
items={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files':items,'raw_results_separate':{'file':'raw_remote_results.tar.gz','bytes':(R/'raw_remote_results.tar.gz').stat().st_size,'sha256':oldraw}}
(R/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
files.add(R/'DELIVERY_MANIFEST.json')
out=R/'ICASSP2027_upgrade_source.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files): z.write(p,'ICASSP2027_upgrade/'+str(p.relative_to(R)))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for f,d in items.items(): assert hashlib.sha256(z.read('ICASSP2027_upgrade/'+f)).hexdigest()==d['sha256']
(R/'PACKAGE_SHA256.txt').write_text(sha(out)+'  '+out.name+'\n')
print(json.dumps({'source_zip_bytes':out.stat().st_size,'files_in_zip':len(files),'figures_checked':len(figs),'source_hash_verified':True,'raw_hash_verified':True}))
