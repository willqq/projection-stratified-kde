"""New-host statistical replay of final/Single, without the legacy MECH loader.

Uses frozen rows, original bandwidth, original preprocessing and sampling seeds.
It does not rerun or replace the official DEANN timing experiment.
"""
from pathlib import Path
import argparse, hashlib, json, pickle, sys
import numpy as np
import pandas as pd
from scipy.io import arff, loadmat
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from projected_kde import ProjectedKDE
p=argparse.ArgumentParser()
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--dataset',choices=['isolet','cifar10','cifar10_gist512','amazon'],required=True)
p.add_argument('--phase',choices=['test','audit','validation'],default='test')
p.add_argument('--queries',type=int,default=200)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
if args.out.exists():raise SystemExit('Output exists; choose a fresh path')
if args.queries<1:raise SystemExit('Positive query count required')
manifest={x['dataset']:x for x in json.loads((ROOT/'data/manifest.json').read_text())}
record=manifest[args.dataset];f=args.data_root/record['relative_path']
if hashlib.sha256(f.read_bytes()).hexdigest()!=record['sha256']:raise SystemExit('Dataset hash mismatch')
if args.dataset=='isolet':raw=pd.read_csv(f,header=None).iloc[:,:-1].to_numpy(dtype=np.float32)
elif args.dataset=='cifar10':
    with f.open('rb') as stream:raw=pickle.load(stream,encoding='bytes')[b'data'].astype(np.float32)
elif args.dataset=='cifar10_gist512':raw=loadmat(f)['X'].astype(np.float32)
else:raw=pd.DataFrame(arff.loadarff(f)[0]).drop(columns=['class_duplicate']).to_numpy(dtype=np.float32)
folder=ROOT/'configs/datasets'/args.dataset
split=json.loads((folder/'split_ids.json').read_text());meta=json.loads((folder/'metadata.json').read_text())
if args.phase=='audit':
    split['audit']=json.loads((ROOT/'configs/audit_ids.json').read_text())['datasets'][args.dataset]['audit_ids']
if not split[args.phase]:raise SystemExit('No frozen queries for this dataset/phase')
ids=split[args.phase][:args.queries]
scaler=MinMaxScaler().fit(raw[split['train']])
tr=scaler.transform(raw[split['train']]).astype(np.float32)
anchor=tr[np.argmax(np.linalg.norm(tr,axis=1))].copy()
transform=lambda x:(2*scaler.transform(x)+anchor).astype(np.float32)
refs=transform(raw[split['reference']]);training=transform(raw[split['train']]);queries=transform(raw[ids])
model=ProjectedKDE(refs,training,meta['bandwidth']);rows=[]
for qi,(rid,q) in enumerate(zip(ids,queries)):
    logs=-cdist(q[None,:].astype(float),model.points,'sqeuclidean')[0]/(2*model.h**2)
    truth=float(logsumexp(logs)-np.log(model.n))
    for M in [32,64,128,256,512]:
        for seed in range(5):
            for single in [False,True]:
                z=model.query(q,M,seed+qi*100003,single)
                rows.append(dict(dataset=args.dataset,phase=args.phase,query_id=qi,raw_row_id=rid,
                                 method='matched_single' if single else 'final',budget=M,seed=seed,
                                 log_truth=truth,log_estimate=z['log_estimate'],
                                 relative_error=abs(np.expm1(z['log_estimate']-truth)),
                                 actual_samples=z['actual_samples']))
args.out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.out,index=False)
print(json.dumps(dict(file=str(args.out),queries=len(ids),rows=len(rows),
                     timing_claim=False,uses_frozen_configuration=True)))
