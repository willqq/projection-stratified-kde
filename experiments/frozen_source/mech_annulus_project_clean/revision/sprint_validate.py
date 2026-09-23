"""Bounded validation screen and prospective shortlist. Never loads formal queries."""
import json,time,argparse,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import core as c
from sprint_common import ROOT,DATASETS,load_data,save_json
from sprint_evaluator import SprintEvaluator
OUT=ROOT/'results/validation';OUT.mkdir(parents=True,exist_ok=True)
CACHE={}

def objects(name):
    if name not in CACHE:
        d=load_data(name);e=SprintEvaluator(d)
        truth=[c.exact_log_mean(d['points'],q,d['old'].h) for q in d['arrays']['validation']]
        CACHE[name]=(d,e,truth)
        save_json(OUT/name/'offline.json',dict(d['offline'],score=e.score.offline_record))
    return CACHE[name]

def run_config(key,cfg,budgets=(128,),control=None):
    control=control if control is not None else (cfg or {}).get('control')
    results=[]
    for name in DATASETS:
        d,e,truth=objects(name);path=OUT/name/(key+'.csv');done=pd.read_csv(path) if path.exists() else pd.DataFrame()
        if cfg is not None and cfg['score']=='projection64':e.ensure_projection()
        identity=dict(config=cfg,control=control,split='validation',sampling_seeds=list(range(5)),query_count=80)
        manifest=path.with_suffix('.identity.json')
        if manifest.exists() and json.loads(manifest.read_text())!=identity:raise RuntimeError('Cached configuration identity changed: '+key)
        if not manifest.exists():save_json(manifest,identity)
        existing=set(done.budget.unique()) if len(done) else set();rows=[]
        for budget0 in existing:
            sub=done[done.budget==budget0]
            if len(sub)!=400 or sub[['query_id','seed']].drop_duplicates().shape[0]!=400:raise RuntimeError('Incomplete cached validation block')
        for budget in budgets:
            if budget in existing:continue
            if cfg is None:
                d['old'].evaluate(key,d['arrays']['validation'][0],budget,np.random.default_rng(987))
            else:e.evaluate(d['arrays']['validation'][0],budget,np.random.default_rng(987),cfg,control)
            for qi,q in enumerate(d['arrays']['validation']):
                for seed in range(5):
                    rng=np.random.default_rng(seed+qi*100003)
                    z=d['old'].evaluate(key,q,budget,rng) if cfg is None else e.evaluate(q,budget,rng,cfg,control)
                    z.update(c.errors(z.pop('log_estimate'),truth[qi]))
                    rows.append(dict(dataset=name,config=key,query_id=qi,raw_row_id=int(d['split']['validation'][qi]),budget=budget,seed=seed,**z))
            print(json.dumps(dict(event='validation',dataset=name,config=key,budget=budget,rows=len(rows))),flush=True)
        if rows:
            new=pd.DataFrame(rows);done=pd.concat([done,new],ignore_index=True) if len(done) else new
            done.to_csv(path,index=False)
        results.append(done)
    return pd.concat(results,ignore_index=True)

def summarize(configs,budgets=(128,)):
    rows=[]
    for name in DATASETS:
        u=pd.read_csv(OUT/name/'uniform_mc.csv');x=pd.read_csv(OUT/name/'exact_annular.csv')
        for key,cfg in configs.items():
            f=pd.read_csv(OUT/name/(key+'.csv'))
            for budget in budgets:
                f0=f[f.budget==budget]
                if len(f0)==0:continue
                eu=u[u.budget==budget].relative_error.mean();ex=x[x.budget==budget].relative_error.mean();ef=f0.relative_error.mean();den=eu-ex
                rec=(eu-ef)/den if den>max(1e-8,.01*eu) else np.nan
                rows.append(dict(dataset=name,config=key,budget=budget,mean_relative_error=ef,MC=eu,exact_annular=ex,recovery=rec,relative_reduction_MC=(eu-ef)/eu,median_time_ms=f0.time_ns.median()/1e6,kernel_evaluations=f0.kernel_evaluations.mean(),**cfg))
    return pd.DataFrame(rows)

def screen():
    for method in ['uniform_mc','exact_annular']:run_config(method,None,(64,128,256))
    configs={}
    for score in ['hamming','real']:
        for J in [4,8]:
            key=f'b0_{score}_J{J}';cfg=dict(score=score,J=J,H_fraction=0.,grouping='quantile',full_target=False,allocation='proportional');configs[key]=cfg;run_config(key,cfg)
    s=summarize(configs);r=s.groupby('config').agg(score=('recovery','median'),latency=('median_time_ms','mean')).sort_values('score',ascending=False)
    top=r.score.max();best=r[r.score>=top-.02].sort_values('latency').index[0];base=configs[best].copy()
    save_json(OUT/'B0_choice.json',dict(selected=best,score=r.to_dict('index'),basis='M128 validation median Recovery; .02-close latency tie'))
    additional=[('a_old_H0',dict(base,grouping='old',full_target=True,H_fraction=0.)),('a_old_Hquarter',dict(base,grouping='old',full_target=True,H_fraction=.25)),('b0a_H0',dict(base,full_target=True,H_fraction=0.)),('b0a_Hquarter',dict(base,full_target=True,H_fraction=.25)),('b0a_Hquarter_neyman',dict(base,full_target=True,H_fraction=.25,allocation='neyman'))]
    for key,cfg in additional:configs[key]=cfg;run_config(key,cfg)
    # Matched H control is recorded for attribution, not silently called novel.
    cfg=dict(base,full_target=True,H_fraction=.25,control='single');configs['matched_Hquarter']=cfg;run_config('matched_Hquarter',cfg,control='single')
    s=summarize(configs);s.to_csv(OUT/'screen_summary.csv',index=False);save_json(OUT/'screen_configs.json',configs)
    print(s.pivot(index='config',columns='dataset',values='mean_relative_error').round(4).to_string(),flush=True)
    print(s.groupby('config').recovery.median().sort_values(ascending=False).to_string(),flush=True)

if __name__=='__main__':screen()
