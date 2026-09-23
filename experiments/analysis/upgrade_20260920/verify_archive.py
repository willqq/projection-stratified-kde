from pathlib import Path
import tarfile,json,hashlib
root=Path(__file__).resolve().parents[1]
archive=root/'raw_remote_results.tar.gz'
with tarfile.open(archive,'r:gz') as t:
 manifest=json.load(t.extractfile('provenance/RESULT_MANIFEST.json'))
# Stream in archive order to avoid repeatedly decompressing the large archive.
verified=[];meta=root/'remote_metadata';meta.mkdir(exist_ok=True)
with tarfile.open(archive,'r|gz') as t:
 for member in t:
  if not member.isfile():continue
  source=t.extractfile(member); h=hashlib.sha256(); n=0
  keep=member.name.startswith('provenance/') or member.name.endswith(('/VERIFICATION.json','/START.json','/DONE.json'))
  destination=meta/member.name
  if keep:destination.parent.mkdir(parents=True,exist_ok=True)
  f=destination.open('wb') if keep else None
  for block in iter(lambda:source.read(1048576),b''):
   h.update(block);n+=len(block)
   if f:f.write(block)
  if f:f.close()
  if member.name in manifest['files']:
   expected=manifest['files'][member.name]
   assert n==expected['bytes'] and h.hexdigest()==expected['sha256'],member.name
   verified.append(member.name)
assert set(verified)==set(manifest['files'])
report=dict(passed=True,verified_result_files=len(verified),archive_bytes=archive.stat().st_size)
(root/'qa/ARCHIVE_VERIFIED.json').write_text(json.dumps(report,indent=2));print(report)
