"""Sprint inputs: frozen old splits, train-fitted preprocessing, guarded audit access."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
import core as c
import run as r
from final_index import IndexedMECH,IndexedEvaluator
ROOT=r.JOBROOT
DATASETS=['isolet','cifar10','cifar10_gist512','amazon']
BUDGETS=[32,64,128,256,512]
SEEDS=[0,1,2,3,4]

def load_data(name,phase='validation'):
    if phase not in ('validation','test','audit'):raise ValueError(phase)
    if phase in ('test','audit') and not (ROOT/'freeze_manifest.json').exists():raise RuntimeError('Formal queries require a frozen method')
    torch.set_num_threads(8)
    meta=json.loads((r.ASSETS/name/'metadata.json').read_text())
    assert r.sha(r.DATA/r.FILES[name])==meta['data_sha256']
    split=json.loads((r.ASSETS/name/'split_ids.json').read_text())
    raw=r.raw_dataset(name)
    active={k:np.asarray(split[k],dtype=np.int64) for k in ('train','reference','validation')}
    if phase=='test':active['test']=np.asarray(split['test'],dtype=np.int64)
    if phase=='audit':
        lock=json.loads((ROOT/'audit/locked_ids.json').read_text())
        active['audit']=np.asarray(lock['datasets'][name]['audit_ids'],dtype=np.int64)
        if len(active['audit'])==0:raise ValueError('No independent audit rows for '+name)
    arrays,scaler,anchor=c.fit_transform_splits(raw,active)
    model=c.source.MultiEncoderContrastiveHash(raw.shape[1],12,5)
    model.load_state_dict(torch.load(r.ASSETS/name/'model.pt',map_location='cpu'));model.eval()
    start=time.perf_counter();index=IndexedMECH(model,arrays['reference'],12,5,2,meta['delta'])
    ev=IndexedEvaluator(arrays['reference'],meta['bandwidth'],meta['radius'],8,index,meta['delta'],2)
    offline={'index_build_seconds':time.perf_counter()-start,'model_training_seconds':meta['train_seconds'],'reference_bytes':ev.points.nbytes,'original_model_sha256':r.sha(r.ASSETS/name/'model.pt'),'data_sha256':meta['data_sha256'],'split_sha256':meta['split_sha256'],'preprocessing_sha256':meta['preprocessing_sha256'],'threads':8}
    return dict(name=name,arrays=arrays,split=active,meta=meta,model=model,index=index,old=ev,points=ev.points,scaler=scaler,anchor=anchor,offline=offline)

def save_json(path,value):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=default))
