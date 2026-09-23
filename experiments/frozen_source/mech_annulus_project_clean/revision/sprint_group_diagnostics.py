"""Validation-only grouping diagnosis; full query distances are offline controls."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
import core as c
import run as r
from final_index import IndexedMECH, IndexedEvaluator
from sprint_scores import ScoreModel, quantile_groups

ROOT = r.JOBROOT
KINDS = ('hamming', 'real')
BUDGETS = (64, 128, 256)
LAYERS = (4, 8)


def corr(a, b, rank=False):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float((spearmanr if rank else pearsonr)(a,b)[0])


def load_validation(name):
    torch.set_num_threads(8)
    folder = ROOT/'assets'/name
    meta = json.loads((folder/'metadata.json').read_text())
    full_split = json.loads((folder/'split_ids.json').read_text())
    split = {k:np.asarray(full_split[k],dtype=np.int64) for k in ('train','reference','validation')}
    # Only these three parts are transformed, never test or locked audit rows.
    arrays, _, _ = c.fit_transform_splits(r.raw_dataset(name),split)
    if r.sha(folder/'model.pt') != meta['model_sha256']:
        raise RuntimeError('Frozen model checksum mismatch')
    model = c.source.MultiEncoderContrastiveHash(arrays['reference'].shape[1],12,5)
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu'))
    model.eval()
    index = IndexedMECH(model,arrays['reference'],12,5,2,meta['delta'])
    evaluator = IndexedEvaluator(arrays['reference'],meta['bandwidth'],meta['radius'],8,index,meta['delta'],2)
    score = ScoreModel(arrays['reference'],arrays['train'],model,index,ROOT/'models'/'score_calibration'/name)
    return arrays,split,meta,evaluator,score


def _variance(cells, values, budget, n):
    counts = c.allocate([len(z) for z in cells],budget)
    return c.predicted_variance([values[z] for z in cells],counts,n),counts


def run_dataset(name, out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/(name+'_DONE.json')).exists():
        raise RuntimeError('Completed diagnosis already exists; do not overwrite')
    start = time.time()
    arrays,split,meta,e,score = load_validation(name)
    (out/(name+'_offline.json')).write_text(json.dumps(score.offline_record,indent=2))
    stats, variances, strata, radius_rows = [],[],[],[]
    n=e.n
    points=e.points
    for qi,q in enumerate(arrays['validation']):
        encoded=score.encode_query(q)
        if encoded['codes']!=e.index.query_codes(q):
            raise AssertionError('Reused code semantics differ from original index')
        Cset,_,old=e.partition(q,encoded['codes'],e.radius)
        C=np.asarray(sorted(Cset),dtype=np.int64)
        old=[np.asarray(x,dtype=np.int64) for x in old if len(x)]
        d2=c.sqdist(points,q)
        logk=-d2/(2*e.h*e.h)
        values=np.exp(logk-np.max(logk)) # shared scale cancels in variance ratios
        qn=float(np.linalg.norm(np.asarray(q,dtype=np.float64)))
        prod=score.reference_norms*qn
        theta=np.arccos(np.clip(np.divide(points@np.asarray(q,dtype=np.float64),prod,
                                  out=np.ones(n),where=prod>0),-1,1))
        thresholds=e.thresholds_for_query(q)
        rings=e.partition(q,encoded['codes'],e.radius)[2]
        union=set()
        for ri,(radius,ring,threshold) in enumerate(zip(np.linspace(0,e.radius,9)[1:],rings,thresholds)):
            union.update(ring)
            app=np.asarray(sorted(union),dtype=np.int64)
            exact=np.flatnonzero(d2<=radius*radius)
            common=np.intersect1d(app,exact,assume_unique=True)
            radius_rows.append(dict(dataset=name,query_id=qi,raw_row_id=int(split['validation'][qi]),
                radius_id=ri,radius=float(radius),radius_over_query_norm=radius/qn if qn else float('inf'),
                angular_bound=c.source.max_angle_from_radius(float(np.linalg.norm(q)),float(radius)),
                threshold=threshold,unique_thresholds=len(set(thresholds)),
                exact_size=len(exact),approx_size=len(app),
                precision=len(common)/len(app) if len(app) else np.nan,
                recall=len(common)/len(exact) if len(exact) else np.nan,
                kernel_mass_recall=float(values[common].sum()/values[exact].sum()) if len(exact) else np.nan))
        base=dict(dataset=name,query_id=qi,raw_row_id=int(split['validation'][qi]),candidate_size=len(C),
                  candidate_ratio=len(C)/n,candidate_kernel_mass_recall=float(values[C].sum()/values.sum()),
                  old_largest_layer_fraction=max([len(z) for z in old],default=0)/max(1,len(C)),
                  old_nonempty_layers=len(old),unique_radius_thresholds=len(set(thresholds)))
        if not len(C):
            stats.append(dict(base,kind='empty'))
            continue
        for kind in KINDS:
            z=score.query(q,C,codes=encoded['codes'],kind=kind,encoded=encoded)
            unique,counts=np.unique(z['raw_score'],return_counts=True)
            stat=dict(base,kind=kind,unique_score_values=len(unique),largest_score_tie_fraction=float(counts.max()/len(C)),
                unique_approximate_distances=len(np.unique(z['d2hat'])),
                d2_spearman=corr(z['d2hat'],d2[C],True),
                raw_score_angle_pearson=corr(z['raw_score'],theta[C]),
                raw_score_angle_spearman=corr(z['raw_score'],theta[C],True),
                calibrated_angle_mae=float(np.mean(np.abs(z['theta_hat']-theta[C]))),
                query_encode_ns=encoded['query_encode_ns'],
                **{k:v for k,v in z.items() if k.endswith('_ns') and k!='query_encode_ns'})
            stats.append(stat)
            for J in LAYERS:
                groups=quantile_groups(C,z['d2hat'],J)
                sizes=[len(x) for x in groups]
                ordered=quantile_groups(C,d2[C],J)
                family=[('approx',groups,-1),('ordered',ordered,-1)]
                for rep in range(5):
                    perm=np.random.default_rng(20260916+qi*1009+rep*100003+J).permutation(C)
                    bounds=np.cumsum([0]+sizes)
                    family.append(('random',[perm[bounds[i]:bounds[i+1]] for i in range(len(sizes))],rep))
                if J==8:
                    family.append(('old_annuli',old,-1))
                for group_kind,cells,rep in family:
                    for si,ids in enumerate(cells):
                        v=values[ids]
                        strata.append(dict(base,kind=kind,J=J,grouping=group_kind,random_repeat=rep,
                            stratum=si,population=len(ids),kernel_scale_log=float(np.max(logk)),
                            scaled_kernel_mean=float(v.mean()),scaled_kernel_variance=float(v.var(ddof=1)) if len(v)>1 else 0.,
                            scaled_kernel_min=float(v.min()),scaled_kernel_max=float(v.max()),
                            exact_distance_mean=float(np.sqrt(d2[ids]).mean()),
                            exact_distance_std=float(np.sqrt(d2[ids]).std())))
                    for M in BUDGETS:
                        variance,allocation=_variance(cells,values,M,n)
                        denominator,_=_variance([C],values,M,n)
                        variances.append(dict(base,kind=kind,J=J,grouping=group_kind,random_repeat=rep,budget=M,
                            variance_ratio=variance/denominator if denominator>0 else np.nan,
                            scaled_predicted_variance=variance,scaled_uniform_variance=denominator,
                            actual_kernel_evaluations=int(allocation.sum()),
                            layer_sizes=json.dumps([len(x) for x in cells]),allocation=json.dumps(allocation.tolist())))
        if (qi+1)%20==0:
            print(json.dumps(dict(dataset=name,validation_queries=qi+1,elapsed_seconds=time.time()-start)),flush=True)
    for suffix,rows in [('query_diagnostics',stats),('variance',variances),('strata',strata),('radii',radius_rows)]:
        pd.DataFrame(rows).to_csv(out/(name+'_'+suffix+'.csv'),index=False)
    vf=pd.DataFrame(variances)
    # Average the five random partitions within query before cross-query summaries.
    vq=vf.groupby(['dataset','kind','J','grouping','budget','query_id'],as_index=False).variance_ratio.mean()
    summary=vq.groupby(['dataset','kind','J','grouping','budget'],as_index=False).agg(
        variance_ratio_median=('variance_ratio','median'),variance_ratio_mean=('variance_ratio','mean'),
        queries=('query_id','size'))
    summary.to_csv(out/(name+'_summary.csv'),index=False)
    sf=pd.DataFrame(stats).groupby(['dataset','kind'],as_index=False).agg(
        unique_scores_median=('unique_score_values','median'),largest_tie_median=('largest_score_tie_fraction','median'),
        d2_spearman_median=('d2_spearman','median'),angle_pearson_median=('raw_score_angle_pearson','median'),
        angle_mae_mean=('calibrated_angle_mae','mean'),candidate_mass_recall_mean=('candidate_kernel_mass_recall','mean'))
    sf.to_csv(out/(name+'_score_summary.csv'),index=False)
    done=dict(dataset=name,queries=len(arrays['validation']),split='validation only',
              transformed_splits=list(arrays),elapsed_seconds=time.time()-start,
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('sprint_scores.py')]})
    (out/(name+'_DONE.json')).write_text(json.dumps(done,indent=2))
    print(summary[summary.budget==128].to_json(orient='records'),flush=True)
    print(sf.to_json(orient='records'),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=list(r.FILES)+['all'],default='all')
    args=parser.parse_args()
    out=ROOT/'results'/'group_diagnostics'
    for name in (list(r.FILES) if args.dataset=='all' else [args.dataset]):
        run_dataset(name,out)

if __name__=='__main__':
    main()
