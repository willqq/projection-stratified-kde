from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import pickle
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import arff
import torch
import core as c

ROOT = Path(__file__).resolve().parents[2]
JOBROOT = ROOT.parent
ASSETS = JOBROOT/'assets'
DATA = ROOT/'data'
FILES = {'isolet':'isolet/isolet1+2+3+4.data',
         'cifar10':'cifar-10-batches-py/data_batch_1',
         'cifar10_gist512':'cifar10-Gist512/Cifar10-Gist512.mat',
         'amazon':'Amazon_initial_50_30_10000/Amazon_initial_50_30_10000.arff'}
MODEL_SEED=20260915
TRAIN={'tables':12,'bits':5,'epochs':24,'batch_size':128,'lr':.001,
       'alpha':1.,'beta':.1,'gamma_balance':.1,'gamma_decorrelation':.1,'device':'cpu'}
METHODS=['exact','uniform_mc','exact_annular','approx_annular_mech']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def git_head(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()


def raw_dataset(name):
    path=DATA/FILES[name]
    if name=='isolet': x=pd.read_csv(path,header=None).iloc[:,:-1].to_numpy(dtype=np.float32)
    elif name=='cifar10':
        with path.open('rb') as f: x=pickle.load(f,encoding='bytes')[b'data'].astype(np.float32)
    elif name=='cifar10_gist512': x=scipy.io.loadmat(path)['X'].astype(np.float32)
    else:
        records,_=arff.loadarff(path)
        x=pd.DataFrame(records).drop(columns=['class_duplicate']).to_numpy(dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all(): raise ValueError('Invalid raw array')
    return x


def setup(name):
    torch.set_num_threads(8)
    raw=raw_dataset(name)
    folder=ASSETS/name
    marker=folder/'metadata.json'
    if marker.exists():
        meta=json.loads(marker.read_text())
        assert meta['data_sha256']==sha(DATA/FILES[name])
        assert meta['training']==TRAIN
        split={k:np.asarray(v,dtype=int) for k,v in json.loads((folder/'split_ids.json').read_text()).items()}
        arrays,_,_=c.fit_transform_splits(raw,split)
        model=c.source.MultiEncoderContrastiveHash(raw.shape[1],TRAIN['tables'],TRAIN['bits'])
        assert sha(folder/'model.pt')==meta['model_sha256']
        model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu'))
        print(json.dumps({'event':'reuse_model','dataset':name,'model_sha256':meta['model_sha256']}),flush=True)
    else:
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'model.pt').exists(): raise RuntimeError('Incomplete asset; investigate before retraining')
        counts=([500,720,80,200] if name=='amazon' else [2800,1800,80,200])
        if sum(counts)>len(raw): raise ValueError('Dataset too small for predefined split')
        perm=np.random.default_rng(MODEL_SEED).permutation(len(raw))
        split={};start=0
        for key,count in zip(['train','reference','validation','test'],counts):
            split[key]=perm[start:start+count];start+=count
        (folder/'split_ids.json').write_text(json.dumps({k:v.tolist() for k,v in split.items()},indent=2))
        arrays,scaler,anchor=c.fit_transform_splits(raw,split)
        np.savez(folder/'preprocessing.npz',scale=scaler.scale_,offset=scaler.min_,anchor=anchor)
        print(json.dumps({'event':'training_start','dataset':name,'shape':list(raw.shape),'training':TRAIN}),flush=True)
        c.source.set_seed(MODEL_SEED)
        torch.use_deterministic_algorithms(True)
        t=time.perf_counter()
        model=c.source.train_mech(arrays['train'],**TRAIN)
        train_seconds=time.perf_counter()-t
        torch.save(model.state_dict(),folder/'model.pt')
        # Validation only; no test query is inspected for radius or bandwidth.
        t=time.perf_counter()
        p64=arrays['reference'].astype(np.float64)
        vd=np.asarray([np.sqrt(c.sqdist(p64,q)) for q in arrays['validation']])
        radius=float(np.median(np.percentile(vd,70,axis=1)))
        bandwidth=.25*float(np.median(np.percentile(vd,25,axis=1)))
        delta=float(np.percentile(np.linalg.norm(arrays['reference'],axis=1),25))/12
        if delta<=0: raise ValueError('Reference norms give zero layer width')
        meta={'dataset':name,'raw_shape':list(raw.shape),'split_counts':dict(zip(split,counts)),
              'data_file':FILES[name],'data_sha256':sha(DATA/FILES[name]),
              'training':TRAIN,'model_seed':MODEL_SEED,'model_sha256':sha(folder/'model.pt'),
              'preprocessing_sha256':sha(folder/'preprocessing.npz'),'split_sha256':sha(folder/'split_ids.json'),
              'train_seconds':train_seconds,'validation_calibration_seconds':time.perf_counter()-t,
              'radius':radius,'bandwidth':bandwidth,'delta':delta,'layers':8,'min_collisions':2,
              'hamming_probe':2,'radius_rule':'median validation-query 70th distance percentile',
              'bandwidth_rule':'.25 * median validation-query 25th distance percentile',
              'preprocessing':'MinMax fitted on training only; common 2*x+training max-norm anchor',
              'code_at_training':git_head(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        marker.write_text(json.dumps(meta,indent=2))
        print(json.dumps({'event':'training_complete','dataset':name,'seconds':train_seconds}),flush=True)
    t=time.perf_counter()
    index=c.source.MECHHashIndex(model,arrays['reference'],TRAIN['tables'],TRAIN['bits'],2)
    evaluator=c.Evaluator(arrays['reference'],meta['bandwidth'],meta['radius'],8,index,meta['delta'],2)
    build_seconds=time.perf_counter()-t
    offline={'dataset':name,'build_seconds':build_seconds,'train_seconds':meta['train_seconds'],
             'point_code_bytes':sum(x.nbytes for x in index.point_codes),
             'index_serialized_bytes':len(pickle.dumps(index,protocol=4)),
             'reference_float64_bytes':evaluator.points.nbytes,
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return arrays,split,meta,evaluator,offline


def benchmark(name,phase,budgets,seeds,nqueries,repeats,outdir):
    arrays,split,meta,evaluator,offline=setup(name)
    (outdir/(name+'_offline.json')).write_text(json.dumps(offline,indent=2))
    which='validation' if phase=='pilot' else 'test'
    queries=arrays[which][:nqueries]
    # Consistent one-query warmup for each method; excluded from result rows.
    for method in METHODS: evaluator.evaluate(method,queries[0],budgets[0],np.random.default_rng(987))
    rows=[];timings=[]
    rawpath=outdir/(name+'_queries.csv')
    timepath=outdir/(name+'_timings.csv')
    if rawpath.exists(): raise RuntimeError('Output already exists; do not overwrite a run')
    for qi,q in enumerate(queries):
        t=time.perf_counter_ns()
        truth=c.exact_log_mean(evaluator.points,q,evaluator.h)
        diagnostic_time=time.perf_counter_ns()-t
        # Rotate order by query to limit drift without selecting by performance.
        order=METHODS[qi%len(METHODS):]+METHODS[:qi%len(METHODS)]
        for method in order:
            for budget in ([0] if method=='exact' else budgets):
                for seed in ([0] if method=='exact' else seeds):
                    results=[]
                    for rep in range(repeats):
                        result=evaluator.evaluate(method,q,max(1,budget),np.random.default_rng(seed+qi*100003))
                        results.append(result)
                        timing={k:v for k,v in result.items() if k.endswith('_ns')}
                        timings.append(dict(dataset=name,query_id=qi,raw_row_id=int(split[which][qi]),
                                            method=method,budget=budget,seed=seed,repeat=rep,**timing))
                    result=results[0].copy()
                    result['time_ns']=float(np.median([r['time_ns'] for r in results]))
                    for key in ['encoding_ns','radius_ns','candidate_ns','allocation_ns','kernel_ns']:
                        result[key]=float(np.median([r[key] for r in results]))
                    result.update(c.errors(result.pop('log_estimate'),truth))
                    result['layer_sizes']=json.dumps(result['layer_sizes'])
                    result['allocation']=json.dumps(result['allocation'])
                    rows.append(dict(dataset=name,phase=phase,query_id=qi,raw_row_id=int(split[which][qi]),
                                     method=method,budget=budget,seed=seed,truth_time_ns=diagnostic_time,**result))
        if (qi+1)%10==0: print(json.dumps({'event':'progress','dataset':name,'phase':phase,'queries':qi+1}),flush=True)
    pd.DataFrame(rows).to_csv(rawpath,index=False)
    pd.DataFrame(timings).to_csv(timepath,index=False)
    c.aggregate(pd.DataFrame(rows)).to_csv(outdir/(name+'_summary.csv'),index=False)
    return len(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['pilot','formal'],required=True)
    parser.add_argument('--dataset',choices=list(FILES),required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    out=JOBROOT/'results'/args.run_id
    out.mkdir(parents=True,exist_ok=True)
    if args.phase=='pilot': budgets,seeds,nq,repeats=[32,128],[0],16,2
    else:
        manifest=json.loads((JOBROOT/'run_manifest.json').read_text())
        assert manifest['frozen'] is True and args.dataset in manifest['datasets']
        budgets,seeds,nq,repeats=(manifest[k] for k in ['budgets','sampling_seeds','test_queries','timing_repeats'])
    command={'argv':sys_argv(),'code_commit':git_head(),'phase':args.phase,
             'budgets':budgets,'sampling_seeds':seeds,'n_queries':nq,'timing_repeats':repeats,
             'threads':8,'device':'CPU','hostname':os.uname().nodename}
    (out/(args.dataset+'_command.json')).write_text(json.dumps(command,indent=2))
    rows=benchmark(args.dataset,args.phase,budgets,seeds,nq,repeats,out)
    (out/(args.dataset+'_DONE.json')).write_text(json.dumps({'rows':rows,'code_commit':git_head(),'completed_at':time.time()}))
    print(json.dumps({'event':'complete','dataset':args.dataset,'rows':rows}),flush=True)


def sys_argv():
    import sys
    return sys.argv

if __name__=='__main__': main()
