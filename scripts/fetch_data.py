"""Download hash-locked author mirrors; original dataset terms still apply."""
from pathlib import Path
import argparse, hashlib, json, urllib.request
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data/raw');a=p.parse_args()
commit='34347a411e7475ebce86f05189e112ad8975e508'
for record in json.loads((ROOT/'data/manifest.json').read_text()):
    dest=a.data_root/record['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        host='https://media.githubusercontent.com/media' if record['dataset']=='cifar10_gist512' else 'https://raw.githubusercontent.com'
        url=f"{host}/jingenyan/mech-annulus-kde/{commit}/data/{record['relative_path']}"
        tmp=dest.with_suffix(dest.suffix+'.part')
        with urllib.request.urlopen(url,timeout=90) as response,tmp.open('wb') as stream:
            while True:
                chunk=response.read(2**20)
                if not chunk:break
                stream.write(chunk)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=record['sha256']:
            raise SystemExit('Downloaded data hash mismatch: '+record['dataset'])
        tmp.rename(dest)
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=record['sha256']:
        raise SystemExit('Existing data hash mismatch: '+record['dataset'])
    print(record['dataset']+': verified',flush=True)
