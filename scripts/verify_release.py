"""Verify the distributed files without requiring experiment dependencies."""
from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parents[1]
records=json.loads((root/'provenance/FILE_MANIFEST.json').read_text())
failures=[]
for name,expected in records.items():
    f=root/name
    if not f.is_file():failures.append({'file':name,'reason':'missing'});continue
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    if h.hexdigest()!=expected:failures.append({'file':name,'reason':'checksum mismatch'})
print(json.dumps({'checked':len(records),'failures':failures},indent=2))
raise SystemExit(bool(failures))
