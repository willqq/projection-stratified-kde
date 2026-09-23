"""Frozen-population offline conditional sampling experiment; no online timing claims."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
BASE=Path('/root/autodl-tmp/icassp_sprint_20260916_0103')
SRC=BASE/'worktree/mech_annulus_project_clean/revision'
sys.path.insert(0,str(SRC))
import core as c
import run as r
from sprint_group_diagnostics import load_validation
from sprint_projection import ProjectionScore
from sprint_estimator import allocate_counts, estimate_strata
from sprint_scores import quantile_groups
OUT=Path('/root/autodl-tmp/icassp_mechanism_20260920')
DS=['isolet','cifar10','cifar10_gist512','amazon']
REPS=1024;SEED=20260920;M=128

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))
def arrsha(x):return hashlib.sha256(np.asarray(x,dtype='<i8').tobytes()).hexdigest()
def cells_from(order,sizes):return list(np.split(order,np.cumsum(sizes)[:-1]))
def positions(sizes,counts,seed):
 rng=np.random.default_rng(seed)
 return [np.stack([rng.choice(int(n),size=int(m),replace=False) for _ in range(REPS)]) if m<n else np.tile(np.arange(n),(REPS,1)) for n,m in zip(sizes,counts)]

def summarize_partition(values,C,H,cells,counts,positions_,uniformvar,meta,order=None):
 sizes=np.array([len(x) for x in cells]);allids=np.concatenate([H]+cells)
 assert np.array_equal(np.sort(allids),np.sort(C)) and len(np.unique(allids))==len(C)
 assert sum(counts)+len(H)==min(M,len(C))
 S=values[C].sum();vs=np.array([values[z].var(ddof=1) if len(z)>1 else 0 for z in cells])
 terms=sizes.astype(float)**2*(1-counts/sizes)*vs/counts
 pred=terms.sum()/S**2
 est=np.full(REPS,values[H].sum()/S)
 for ids,n,idx in zip(cells,sizes,positions_):est+=n*values[ids][idx].mean(axis=1)/S
 empirical=est.var(ddof=1)
 row=dict(meta,H_count=len(H),candidate_size=len(C),candidate_hash=arrsha(np.sort(C)),H_hash=arrsha(H),
  group_sizes=json.dumps(sizes.tolist()),allocation=json.dumps(counts.tolist()),J=len(cells),
  actual_samples=int(len(H)+sum(counts)),predicted_relative_variance=pred,
  empirical_relative_variance=empirical,predicted_variance_ratio=pred/uniformvar if uniformvar>0 else np.nan,
  empirical_over_predicted=empirical/pred if pred>0 else np.nan,
  mean_relative_error=float(np.abs(est-1).mean()),relative_bias=float(est.mean()-1),
  weighted_within_variance=float(np.sum(sizes*vs)/sizes.sum()),
  H_mass_recall=float(values[H].sum()/S),
  ordering_spearman=float(spearmanr(np.arange(len(order)),meta_dist[order]).statistic) if order is not None else np.nan)
 strata=[dict(meta,stratum=j,population=int(len(ids)),samples=int(counts[j]),scaled_kernel_mean=float(values[ids].mean()),
  scaled_kernel_variance=float(vs[j]),predicted_relative_variance_contribution=float(terms[j]/S**2)) for j,ids in enumerate(cells)]
 return row,strata,est

def start(stage):
 path=OUT/'results'/stage;path.mkdir(parents=True,exist_ok=False)
 save(path/'START.json',dict(unix=time.time(),protocol_sha256=sha(OUT/'EXPERIMENT_ROADMAP.md'),script_sha256=sha(__file__),
  python=platform.python_version(),numpy=np.__version__,repetitions=REPS,seed=SEED,
  scientific_source_sha256={x:sha(SRC/x) for x in ['core.py','sprint_group_diagnostics.py','sprint_projection.py','sprint_scores.py','sprint_estimator.py']}))
 return path

def fixed():
 global meta_dist
 path=start('fixed_union');qa=[]
 for di,ds in enumerate(DS):
  begin=time.time();arrays,split,meta,ev,score=load_validation(ds)
  proj=ProjectionScore(arrays['reference'],arrays['train'],BASE/'models/projection64'/ds)
  rows=[];strata=[];raw={};cached={}
  for qi,q in enumerate(arrays['validation']):
   encoded=score.encode_query(q);Cset,_,old=ev.partition(q,encoded['codes'],ev.radius)
   C=np.array(sorted(Cset),dtype=np.int64);old=[np.array(x,dtype=np.int64) for x in old if len(x)]
   assert len(C)>=128
   d2=c.sqdist(ev.points,q);meta_dist=d2;logk=-d2/(2*ev.h**2);values=np.exp(logk-logk.max())
   projected=proj.query(q)['d2hat'];porder=np.lexsort((np.arange(ev.n),projected));H=porder[:32];rem=porder[32:]
   ham=score.query(q,C,codes=encoded['codes'],kind='hamming',encoded=encoded)['d2hat']
   cached.update({f'q{qi}_values':values,f'q{qi}_d2':d2,f'q{qi}_projected':projected,f'q{qi}_C':C,f'q{qi}_H':H,
                 f'q{qi}_raw_row_id':np.array(split['validation'][qi]),f'q{qi}_log_kernel_scale':np.array(logk.max())})
   common=dict(dataset=ds,query_id=qi,raw_row_id=int(split['validation'][qi]),h=ev.h,budget=M,
               repetitions=REPS,kernel_scale_log=float(logk.max()),population_normalizer=ev.n)
   blocks=[('old_sizes',C,np.empty(0,dtype=int),[len(z) for z in old]),
           ('equal_C',C,np.empty(0,dtype=int),[len(z) for z in np.array_split(C,8)]),
           ('matched_H',np.arange(ev.n),H,[len(z) for z in np.array_split(rem,8)])]
   for bi,(block,pop,hids,sizes) in enumerate(blocks):
    remainder=np.setdiff1d(pop,hids);counts=allocate_counts(sizes,M-len(hids));S=values[pop].sum()
    uv=(len(remainder)**2*(1-(M-len(hids))/len(remainder))*values[remainder].var(ddof=1)/(M-len(hids)))/S**2
    pos=positions(sizes,counts,SEED+di*10000000+qi*1000+bi*100)
    po=remainder[np.lexsort((remainder,projected[remainder]))];xo=remainder[np.lexsort((remainder,d2[remainder]))]
    family=[('projected',-1,cells_from(po,sizes),po),('exact_order',-1,cells_from(xo,sizes),xo)]
    if block=='old_sizes':family.append(('original_norm_angle',-1,old,None))
    if block=='equal_C':
     ho=C[np.lexsort((C,ham))];family.append(('norm_angle_hamming_quantiles',-1,cells_from(ho,sizes),ho))
    for rep in range(5):
     ro=np.random.default_rng(SEED+di*10000000+qi*1000+bi*100+50+rep).permutation(remainder)
     family.append(('random',rep,cells_from(ro,sizes),ro))
    exactball=np.flatnonzero(d2<=ev.radius**2);inter=np.intersect1d(pop,exactball)
    bm=dict(common,block=block,candidate_fraction=len(pop)/ev.n,candidate_kernel_mass_recall=float(values[pop].sum()/values.sum()),
      identifier_recall=len(inter)/len(exactball) if len(exactball) else np.nan,
      target_scaled=float(S/ev.n),uniform_relative_variance=uv)
    for kind,rep,cells,order in family:
     m=dict(bm,partition=kind,partition_repeat=rep)
     row,ss,est=summarize_partition(values,pop,hids,cells,counts,pos,uv,m,order)
     rows.append(row);strata.extend(ss);key=f'{qi}_{block}_{kind}_{rep}'
     raw[key]=est;cached[key+'_partition']=np.concatenate(cells);cached[key+'_sizes']=np.array(sizes)
    single_counts=np.array([M-len(hids)])
    single_pos=positions([len(remainder)],single_counts,SEED+di*10000000+qi*1000+bi*100+99)
    m=dict(bm,partition='single',partition_repeat=-1)
    row,ss,est=summarize_partition(values,pop,hids,[remainder],single_counts,single_pos,uv,m)
    rows.append(row);strata.extend(ss);raw[f'{qi}_{block}_single_-1']=est
    if qi<2:
     # Cross-check actual unchanged estimator with identical sample positions and original log kernels.
     seed=6000+qi+bi*10;rng=np.random.default_rng(seed)
     sample=np.concatenate([hids]+[cell[rng.choice(len(cell),int(m),replace=False)] for cell,m in zip(cells_from(po,sizes),counts)])
     manual=(values[hids].sum()+sum(len(cell)*values[ids].mean() for cell,ids in zip(cells_from(po,sizes),np.split(sample[len(hids):],np.cumsum(counts)[:-1]))))/S
     mapids=np.sort(pop);localH=np.searchsorted(mapids,hids);localcells=[np.searchsorted(mapids,z) for z in cells_from(po,sizes)]
     z=estimate_strata(len(pop),localH,localcells,M,np.random.default_rng(seed),lambda ix:logk[mapids[ix]],validate=True)
     actual=float(np.exp(z['log_estimate']-(np.log(S)+logk.max()-np.log(len(pop)))))
     assert abs(manual-actual)<1e-10,(manual,actual)
     qa.append(dict(dataset=ds,query_id=qi,block=block,manual=manual,production=actual,abs_diff=abs(manual-actual)))
   if (qi+1)%20==0:print(json.dumps(dict(stage='fixed_union',dataset=ds,queries=qi+1,seconds=time.time()-begin)),flush=True)
  pd.DataFrame(rows).to_csv(path/f'{ds}_query_metrics.csv',index=False)
  pd.DataFrame(strata).to_csv(path/f'{ds}_strata.csv',index=False)
  np.savez_compressed(path/f'{ds}_estimates.npz',**raw)
  np.savez_compressed(path/f'{ds}_cache.npz',**cached)
  save(path/f'{ds}_DONE.json',dict(queries=80,rows=len(rows),seconds=time.time()-begin,rawids=split['validation'],meta=meta))
 save(path/'CORRECTNESS.json',qa);save(path/'DONE.json',dict(unix=time.time(),datasets=DS))

def degrade():
 global meta_dist
 assert (OUT/'ACTIVATION.md').exists()
 path=start('degradation')
 for di,ds in enumerate(DS):
  begin=time.time();cache=np.load(OUT/'results/fixed_union'/f'{ds}_cache.npz');rows=[];strata=[];raw={};partitions={}
  for qi in range(80):
   values=cache[f'q{qi}_values'];d2=cache[f'q{qi}_d2'];meta_dist=d2;H=cache[f'q{qi}_H'];pop=np.arange(len(values));rem=np.setdiff1d(pop,H)
   exact=rem[np.lexsort((rem,d2[rem]))];sizes=[len(z) for z in np.array_split(rem,8)];counts=allocate_counts(sizes,96)
   S=values.sum();uv=len(rem)**2*(1-96/len(rem))*values[rem].var(ddof=1)/96/S**2
   pos=positions(sizes,counts,SEED+di*10000000+qi*1000+888)
   for rep in range(5):
    rng=np.random.default_rng(SEED+di*10000000+qi*1000+rep+700);pathorder=rng.permutation(len(rem))
    for level in [0,.25,.5,.75,1]:
     order=exact.copy();sel=pathorder[:int(level*len(rem))]
     order[sel]=np.random.default_rng(SEED+di*10000000+qi*1000+rep*10+int(level*4)+900).permutation(order[sel])
     cells=cells_from(order,sizes)
     meta=dict(dataset=ds,query_id=qi,raw_row_id=int(cache[f'q{qi}_raw_row_id']),block='matched_H',partition='degradation',partition_repeat=rep,
       level=level,budget=M,repetitions=REPS,uniform_relative_variance=uv,candidate_fraction=1.,candidate_kernel_mass_recall=1.)
     row,ss,est=summarize_partition(values,pop,H,cells,counts,pos,uv,meta,order)
     rows.append(row);strata.extend(ss);raw[f'{qi}_{rep}_{level}']=est;partitions[f'{qi}_{rep}_{level}']=order
  pd.DataFrame(rows).to_csv(path/f'{ds}_query_metrics.csv',index=False);pd.DataFrame(strata).to_csv(path/f'{ds}_strata.csv',index=False)
  np.savez_compressed(path/f'{ds}_estimates.npz',**raw);np.savez_compressed(path/f'{ds}_partitions.npz',**partitions)
  save(path/f'{ds}_DONE.json',dict(queries=80,rows=len(rows),seconds=time.time()-begin))
  print(json.dumps(dict(stage='degradation',dataset=ds,seconds=time.time()-begin)),flush=True)
 save(path/'DONE.json',dict(unix=time.time(),datasets=DS))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fixed','degrade']);args=ap.parse_args()
 fixed() if args.stage=='fixed' else degrade()
