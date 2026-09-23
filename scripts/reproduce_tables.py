"""Recompute manuscript Tables 1/2 from immutable per-query observations.

No estimator is rerun and no fresh latency is invented. Bootstrap clusters are
queries; the five sampling seeds are averaged before resampling. Historical
bootstrap order and RNG are preserved for the printed intervals.
"""
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
NAMES={'isolet':'ISOLET','cifar10':'CIFAR-10-Small','cifar10_gist512':'GIST-512','amazon':'Amazon'}
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/paper');args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
# RQ4 uses the stronger validation-tuned DEANN experiment, not the old baseline.
frames=[]
for f in sorted((ROOT/'results/paper_inputs/deann_formal').rglob('*_queries.csv.gz')):
    dataset,phase=f.parts[-3:-1];z=pd.read_csv(f);z['dataset']=dataset;z['phase']=phase;frames.append(z)
raw=pd.concat(frames,ignore_index=True)
summary=raw.groupby(['dataset','phase','method','budget'],as_index=False).agg(error=('relative_error','mean'),latency_ms=('time_ns',lambda x:x.median()/1e6))
summary.to_csv(args.out/'latency_points.csv',index=False)
lines=[r'\begin{tabular}{lccccc}',r'\toprule',r'Dataset & Uniform MC & Exact annular & Projected strata & DEANN-E & Exact time \\',r'\midrule']
for ds,name in NAMES.items():
    g=summary[(summary.dataset==ds)&(summary.phase=='test')];cells=[]
    for m in ['uniform_mc','exact_annular','final','deann_mean_relative_error']:
        a=g[(g.method==m)&(g.budget==128)]
        if len(a)==0 and m=='deann_mean_relative_error':a=g[(g.method=='deann_original')&(g.budget==128)]
        assert len(a)==1,(ds,m)
        r=a.iloc[0];cells.append(f'{100*r.error:.1f} / {r.latency_ms:.3f}')
    exact=g[g.method=='exact'];assert len(exact)==1
    lines.append(name+' & '+' & '.join(cells)+f' & {exact.iloc[0].latency_ms:.3f} '+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(args.out/'table1.tex').write_text('\n'.join(lines)+'\n')
# Table 2 retains the original formal matched-H observations and bootstrap.
frames=[pd.read_csv(f,usecols=['dataset','phase','method','budget','query_id','raw_row_id','seed','relative_error']) for f in sorted((ROOT/'results/paper_inputs/sprint').rglob('basic_queries.csv.gz'))]
sprint=pd.concat(frames,ignore_index=True)
q=sprint.groupby(['phase','dataset','method','budget','query_id','raw_row_id'],as_index=False).relative_error.mean()
rng=np.random.default_rng(20260916);contrasts={}
for (phase,ds,M),g in q[q.budget>0].groupby(['phase','dataset','budget']):
    w=g.pivot(index='raw_row_id',columns='method',values='relative_error').sort_index();assert len(w)==200
    ix=rng.integers(0,len(w),size=(1000,len(w)))
    a=w['final'].to_numpy();b=w['matched_single'].to_numpy()
    gain=100*(1-a.mean()/b.mean());bs=100*(1-a[ix].mean(axis=1)/b[ix].mean(axis=1))
    contrasts[(phase,ds,M)]=dict(Single=100*b.mean(),Multi=100*a.mean(),gain=gain,ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975))
v=pd.concat([pd.read_csv(f) for f in sorted((ROOT/'results/paper_inputs/fixed_c').rglob('*_variance.csv.gz'))],ignore_index=True)
v=v.groupby(['phase','dataset','query_id','grouping','budget'],as_index=False).variance_ratio.mean()
v=v.groupby(['phase','dataset','grouping','budget']).variance_ratio.median()
lines=[r'\begin{tabular}{lccccccc}',r'\toprule',r'& \multicolumn{3}{c}{A: fixed-population variance ratio} & \multicolumn{4}{c}{B: matched-$H$ remainder comparison} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-8}',r'Dataset / queries & Projected & Random & Ordered & Single (\%) & Multi (\%) & Gain (\%) & 95\% CI \\',r'\midrule']
ledger=[]
for phase in ['test','audit']:
    if phase=='audit':lines.append(r'\midrule')
    for ds,name in NAMES.items():
        if phase=='audit' and ds=='amazon':continue
        t=contrasts[(phase,ds,128)];vv=[v.loc[(phase,ds,g,128)] for g in ['approx','random','ordered']]
        vals=[f'{x:.3f}' for x in vv]+[f'{t[k]:.1f}' for k in ['Single','Multi','gain']]+[f"[{t['ci_low']:.1f}, {t['ci_high']:.1f}]"]
        lines.append(name+' / '+phase+' & '+' & '.join(vals)+' '+r'\\')
        ledger.append(dict(dataset=ds,phase=phase,**t,projected_variance=vv[0],random_variance=vv[1],ordered_variance=vv[2]))
lines += [r'\bottomrule',r'\end{tabular}'];(args.out/'table2.tex').write_text('\n'.join(lines)+'\n')
pd.DataFrame(ledger).to_csv(args.out/'table2_values.csv',index=False)
checks={}
for number in [1,2]:
    original=(ROOT/f'paper/tables/table{number}_final.tex').read_text()
    generated=(args.out/f'table{number}.tex').read_text()
    checks[f'table{number}_matches_manuscript']=re.sub(r'\s+','',original)==re.sub(r'\s+','',generated)
(args.out/'TABLE_CHECK.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if not all(checks.values()):raise SystemExit('Mismatch: inspect outputs; do not alter observations to force a match.')
