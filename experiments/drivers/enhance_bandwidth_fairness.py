"""Predeclared evidence-only experiments. Frozen scientific modules are imported unchanged."""
import argparse, copy, hashlib, json, math, platform, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import core as c, run as r
from sprint_common import ROOT, DATASETS, load_data, save_json
from sprint_evaluator import SprintEvaluator
from sprint_estimator import allocate_counts
from sprint_projection import ProjectionScore
import deann_adapter as da

OUT=ROOT/'enhancement'
PRIOR=Path('/root/autodl-tmp/icassp_sprint_20260916_0103')
CFG=dict(score='projection64',J=8,H_fraction=.25,grouping='quantile',full_target=True,allocation='proportional')
MS=[32,64,128,256,512]
HM=[.5,.75,1.,1.5,2.]
NS=[1000,1800,5000,10000,20000]

def emit(**x):
    print(json.dumps(dict(unix=time.time(),**x)),flush=True)

def setup(stage):
    OUT.mkdir(exist_ok=True)
    p=OUT/(stage+'_START.json')
    if p.exists(): raise RuntimeError('Stage already started: '+stage)
    save_json(p,dict(unix=time.time(),stage=stage,protocol_sha=r.sha(ROOT/'EXPERIMENT_DECISION.md'),
        script_sha=r.sha(Path(__file__)),head=r.git_head(),python=platform.python_version(),
        processes=subprocess.check_output(['ps','-eo','pid,pcpu,comm,args'],text=True)))

def evaluator(data):
    e=SprintEvaluator(data);e.ensure_projection();return e

def control(e):
    z=c.Evaluator(e.points,e.h,e.old.radius,8);z.points=e.points;return z

def phases(name):return ['test','audit'] if name!='amazon' else ['test']

def variance(values,cells,budget,n):
    counts=allocate_counts([len(z) for z in cells],budget)
    return c.predicted_variance([values[z] for z in cells],counts,n)

def bandwidth():
    setup('bandwidth');out=OUT/'bandwidth';out.mkdir()
    replay=0.;nrows=0
    for name in DATASETS:
      for phase in phases(name):
        data=load_data(name,phase);e=evaluator(data);h0=e.h;qset=data['arrays'][phase]
        old=pd.read_csv(PRIOR/'results/formal'/phase/name/'basic_queries.csv')
        old=old[old.budget==128].set_index(['method','query_id','seed'])
        rows=[];vr=[]
        for qi,q in enumerate(qset):
          d2=c.sqdist(e.points,q) # OFFLINE oracle; never passed to an online estimator.
          p=e.prepare(q,128,CFG);H=p['H'];cells=p['cells'];rem=np.concatenate(cells)
          assert len(np.unique(np.concatenate([H,rem])))==e.n and len(H)==32
          Hhash=hashlib.sha256(H.tobytes()).hexdigest()
          for factor in HM:
            e.h=h0*factor;cc=control(e)
            logs=-d2/(2*e.h**2);truth=float(c.logsumexp(logs)-math.log(e.n))
            values=np.exp(logs-logs.max());fscale=values.mean()
            vm=variance(values,cells,96,e.n);vs=variance(values,[rem],96,e.n)
            vu=variance(values,[e.id_array],128,e.n)
            vr.append(dict(dataset=name,phase=phase,query_id=qi,raw_row_id=int(data['split'][phase][qi]),
                h_factor=factor,budget=128,H_hash=Hhash,H_mass_recall=values[H].sum()/values.sum(),
                single_relative_variance=vs/fscale**2,multi_relative_variance=vm/fscale**2,
                uniform_relative_variance=vu/fscale**2,variance_ratio=vm/vs if vs>0 else np.nan))
            for method in ['uniform_mc','exact_annular','matched_single','final']:
              for seed in range(5):
                rng=np.random.default_rng(seed+qi*100003)
                if method in ['uniform_mc','exact_annular']:z=cc.evaluate(method,q,128,rng)
                else:
                  z=e.evaluate(q,128,rng,CFG,control='single' if method=='matched_single' else None,keep_ids=True)
                  assert z['H_ids']==H.tolist() and len(set(z['selected_ids']))==128
                  for key in ['H_ids','candidate_ids','selected_ids']:z.pop(key)
                assert z['actual_samples']==z['kernel_evaluations']==128
                z.update(c.errors(z.pop('log_estimate'),truth))
                if factor==1:
                  diff=abs(z['relative_error']-old.loc[(method,qi,seed),'relative_error'])
                  replay=max(replay,diff);assert diff<1e-11,(name,phase,method,qi,diff)
                rows.append(dict(dataset=name,phase=phase,query_id=qi,raw_row_id=int(data['split'][phase][qi]),
                    h_factor=factor,h=e.h,budget=128,method=method,seed=seed,H_hash=Hhash,**z))
        pd.DataFrame(rows).to_csv(out/f'{name}_{phase}_queries.csv',index=False)
        pd.DataFrame(vr).to_csv(out/f'{name}_{phase}_variance.csv',index=False)
        nrows+=len(rows);emit(stage='bandwidth',dataset=name,phase=phase,rows=len(rows),replay_max=replay)
    save_json(out/'DONE.json',dict(rows=nrows,h0_max_error_difference=replay,completed_unix=time.time()))

def tune(e,queries,out):
    out.mkdir(parents=True,exist_ok=False)
    ann,off=da.build(e.points);save_json(out/'offline.json',off)
    truth=[c.exact_log_mean(e.points,q,e.h) for q in queries]
    probes=sorted(set([1,2,4,8,16,off['nlist']]))
    rows=[];agg=[]
    for probe in probes:
      for frac in [.125,.25,.5,.75]:
        subset=[]
        for m in [64,128,256]:
          for seed in [0,1]:
            est,k,mm,fit=da.make_estimator(e.points,e.h,ann,m,frac,probe,seed)
            da.evaluate(est,ann,queries[0],k,mm)
            for qi,q in enumerate(queries):
              reps=[da.evaluate(est,ann,q,k,mm) for _ in range(3)]
              z=c.errors(reps[0]['log_estimate'],truth[qi])
              row=dict(nprobe=probe,near_fraction=frac,budget=m,seed=seed,query_id=qi,
                       time_ns=float(np.median([x['time_ns'] for x in reps])),**z)
              rows.append(row);subset.append(row)
        err=np.mean([x['relative_error'] for x in subset]);lat=np.median([x['time_ns'] for x in subset])
        agg.append(dict(nprobe=probe,near_fraction=frac,mean_relative_error=err,median_time_ns=lat,error_time_product=err*lat))
        emit(stage='validation',out=str(out),nprobe=probe,fraction=frac)
    a=pd.DataFrame(agg);a.to_csv(out/'grid.csv',index=False);pd.DataFrame(rows).to_csv(out/'query_level.csv',index=False)
    anchors=[];seen=set()
    for criterion in ['mean_relative_error','median_time_ns','error_time_product']:
      keys=list(dict.fromkeys([criterion,'median_time_ns','nprobe','near_fraction']))
      z=a.sort_values(keys).iloc[0];pair=(int(z.nprobe),float(z.near_fraction))
      if pair not in seen:
        anchors.append(dict(nprobe=pair[0],near_fraction=pair[1],selected_by=criterion));seen.add(pair)
    freeze=dict(created_unix=time.time(),n=e.n,nlist=off['nlist'],anchors=anchors,grid_sha256=r.sha(out/'grid.csv'),
                protocol_sha256=r.sha(ROOT/'EXPERIMENT_DECISION.md'),budgets=[64,128,256],validation_queries=len(queries),sampling_seeds=[0,1])
    save_json(out/'selection.json',freeze);return freeze

def deann_validation():
    setup('deann_validation');selected={}
    for name in DATASETS:
      data=load_data(name,'validation');e=evaluator(data)
      selected[name]=tune(e,data['arrays']['validation'],OUT/'deann_validation'/name)
    save_json(OUT/'DEANN_FROZEN.json',dict(created_unix=time.time(),datasets=selected))

def formal(e,queries,rawids,variants,out,budgets,old_replay=None):
    out.mkdir(parents=True,exist_ok=False);cc=control(e);rows=[];times=[];maxdiff=0.
    truths=[c.exact_log_mean(e.points,q,e.h) for q in queries]
    offline=dict(projection=e.projection.offline_record,n=e.n,feature_dimension=e.points.shape[1],h=e.h,R=e.old.radius)
    pd.DataFrame(dict(query_id=np.arange(len(queries)),raw_row_id=rawids,log_truth=truths)).to_csv(out/'truth.csv',index=False)
    old=None
    if old_replay:
      old=pd.read_csv(old_replay/'basic_queries.csv').set_index(['method','query_id','budget','seed'])
    methods=['exact','uniform_mc','exact_annular','final']
    def one(method,q,m,seed,qi):
      rng=np.random.default_rng(seed+qi*100003)
      if method=='final':
        z=e.evaluate(q,m,rng,CFG,keep_ids=False)
        assert z['actual_samples']==z['kernel_evaluations']==m
        assert z['H_count']==m//4 and z['candidate_size']==e.n
      else:z=cc.evaluate(method,q,max(1,m),rng)
      return z
    for method in methods:one(method,queries[0],32,987,0)
    for qi,q in enumerate(queries):
      rot=qi%4
      for method in methods[rot:]+methods[:rot]:
        for m in ([0] if method=='exact' else budgets):
          for seed in ([0] if method=='exact' else range(5)):
            reps=[one(method,q,m,seed,qi) for _ in range(3)];z=reps[0].copy()
            for zz in reps[1:]:assert zz['log_estimate']==z['log_estimate']
            for key in list(z):
              if key.endswith('_ns'):z[key]=float(np.median([rr[key] for rr in reps]))
            z.update(c.errors(z.pop('log_estimate'),truths[qi]))
            if old is not None:
              diff=abs(z['relative_error']-old.loc[(method,qi,m,seed),'relative_error'])
              maxdiff=max(maxdiff,diff);assert diff<1e-11,(method,qi,m,seed,diff)
            base=dict(query_id=qi,raw_row_id=int(rawids[qi]),method=method,budget=m,seed=seed)
            rows.append(dict(base,**z))
            for rep,rr in enumerate(reps):times.append(dict(base,repeat=rep,**{k:v for k,v in rr.items() if k.endswith('_ns')}))
    pd.DataFrame(rows).to_csv(out/'basic_queries.csv',index=False);pd.DataFrame(times).to_csv(out/'basic_timings.csv',index=False)
    emit(stage='formal_basic',out=str(out),rows=len(rows),replay_max=maxdiff)
    ann,annoff=da.build(e.points);offline['ann']=annoff
    rows=[];times=[];fits=[];old_deann=None;deann_replay=0.
    if old_replay:old_deann=pd.read_csv(old_replay/'deann_queries.csv').set_index(['query_id','budget','seed'])
    for label,dcfg in variants.items():
      for m in budgets:
        for seed in range(5):
          est,k,mm,fit=da.make_estimator(e.points,e.h,ann,m,dcfg['near_fraction'],dcfg['nprobe'],seed)
          fits.append(dict(method=label,budget=m,seed=seed,fit_seconds=fit))
          da.evaluate(est,ann,queries[0],k,mm)
          for qi,q in enumerate(queries):
            reps=[da.evaluate(est,ann,q,k,mm) for _ in range(3)];z=reps[0].copy()
            for key in list(z):
              if key.endswith('_ns'):z[key]=float(np.median([rr[key] for rr in reps]))
            z.update(c.errors(z.pop('log_estimate'),truths[qi]))
            if label=='deann_original' and old_deann is not None:
              diff=abs(z['relative_error']-old_deann.loc[(qi,m,seed),'relative_error'])
              deann_replay=max(deann_replay,diff);assert diff<1e-9,(qi,m,seed,diff)
            base=dict(query_id=qi,raw_row_id=int(rawids[qi]),method=label,budget=m,seed=seed,nprobe=dcfg['nprobe'],near_fraction=dcfg['near_fraction'])
            rows.append(dict(base,**z))
            for rep,rr in enumerate(reps):times.append(dict(base,repeat=rep,**{k:v for k,v in rr.items() if k.endswith('_ns')}))
      emit(stage='formal_deann',out=str(out),variant=label)
    pd.DataFrame(rows).to_csv(out/'deann_queries.csv',index=False);pd.DataFrame(times).to_csv(out/'deann_timings.csv',index=False)
    pd.DataFrame(fits).to_csv(out/'fits.csv',index=False);save_json(out/'offline.json',offline)
    save_json(out/'DONE.json',dict(completed_unix=time.time(),variants=variants,basic_replay_max=maxdiff,deann_replay_max=deann_replay,queries=len(queries),budgets=budgets,sampling_seeds=list(range(5))))

def variants(freeze,oldcfg):
    result={'deann_original':oldcfg};seen={(oldcfg['nprobe'],oldcfg['near_fraction'])}
    for z in freeze['anchors']:
      pair=(z['nprobe'],z['near_fraction'])
      if pair not in seen:result['deann_'+z['selected_by']]=z;seen.add(pair)
    return result

def deann_formal():
    assert (OUT/'DEANN_FROZEN.json').exists();setup('deann_formal')
    frozen=json.loads((OUT/'DEANN_FROZEN.json').read_text())
    base=json.loads((ROOT/'final_config.yaml').read_text())
    for name in DATASETS:
      vs=variants(frozen['datasets'][name],base['datasets'][name]['deann'])
      for phase in phases(name):
        data=load_data(name,phase);e=evaluator(data)
        formal(e,data['arrays'][phase],data['split'][phase],vs,OUT/'deann_formal'/name/phase,MS,PRIOR/'results/formal'/phase/name)

def scaling():
    setup('scaling');name='cifar10_gist512';data=load_data(name,'validation');template=evaluator(data)
    raw=r.raw_dataset(name);split=json.loads((r.ASSETS/name/'split_ids.json').read_text())
    audits=json.loads((ROOT/'audit/locked_ids.json').read_text())['datasets'][name]['audit_ids']
    excluded=set(audits)
    for ids in split.values():
      if isinstance(ids,list):excluded.update(ids)
    pool=np.array([i for i in range(len(raw)) if i not in excluded],dtype=np.int64)
    np.random.default_rng(20260920).shuffle(pool)
    order=np.concatenate([np.array(split['reference'],dtype=np.int64),pool])[:max(NS)]
    qids=np.array(audits[:100],dtype=np.int64)
    assert len(order)==max(NS) and len(set(order))==max(NS) and not set(order)&set(qids)
    save_json(OUT/'scaling_ids.json',dict(created_unix=time.time(),sizes=NS,reference_order=order,query_ids=qids,validation_ids=split['validation'],raw_sha256=r.sha(r.DATA/r.FILES[name]),seed=20260920))
    qarr=(2*data['scaler'].transform(raw[qids])+data['anchor']).astype(np.float32)
    allrefs=(2*data['scaler'].transform(raw[order])+data['anchor']).astype(np.float32)
    assert np.array_equal(allrefs[:1800],data['arrays']['reference'])
    oldcfg=json.loads((ROOT/'final_config.yaml').read_text())['datasets'][name]['deann']
    for n in NS:
      e=copy.copy(template);e.points=np.ascontiguousarray(allrefs[:n],dtype=np.float64);e.n=n;e.id_array=np.arange(n,dtype=np.int64)
      e.projection=ProjectionScore(allrefs[:n],data['arrays']['train'],ROOT/'models'/('scaling_'+str(n)))
      freeze=tune(e,data['arrays']['validation'],OUT/'scaling_validation'/str(n))
      vs=variants(freeze,oldcfg)
      formal(e,qarr,qids,vs,OUT/'scaling'/str(n),[64,128,256])
    save_json(OUT/'scaling_DONE.json',dict(completed_unix=time.time(),sizes=NS))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['bandwidth','deann_validation','deann_formal','scaling']);a=ap.parse_args()
    globals()[a.stage]()
