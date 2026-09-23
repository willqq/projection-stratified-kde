from pathlib import Path
import argparse, hashlib, json
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[1];results=[]
for row in json.loads((root/'data/manifest.json').read_text()):
    f=args.data_root/row['relative_path']
    h=hashlib.sha256()
    if f.exists():
        with f.open('rb') as stream:
            for b in iter(lambda:stream.read(2**20),b''):h.update(b)
    results.append(dict(dataset=row['dataset'],exists=f.exists(),
                        sha256_matches=f.exists() and h.hexdigest()==row['sha256']))
print(json.dumps(results,indent=2))
raise SystemExit(0 if all(x['sha256_matches'] for x in results) else 1)
