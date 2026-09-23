"""Predeclared sensitivity check; no parameter selection and no primary-result overwrite."""
import sys, json, time, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import core as c
from sprint_common import load_data, save_json
from sprint_evaluator import SprintEvaluator
import sprint_projection as sp

SEEDS = [20260916, 20260917, 20260918]
BUDGETS = [64, 128, 256]
CFG = dict(score='projection64', J=8, H_fraction=.25, grouping='quantile', full_target=True, allocation='proportional')
OUT = ROOT/'seed_sensitivity'
PRIOR = Path('/root/autodl-tmp/icassp_sprint_20260916_0103')

def main():
    OUT.mkdir(exist_ok=False)
    protocol = dict(created_unix=time.time(), projection_seeds=SEEDS, budgets=BUDGETS,
        sampling_seeds=list(range(5)), queries_per_phase=200, cfg=CFG,
        selection='None: all seeds reported; original manuscript seed unchanged',
        phases='All four existing test sets and three previously unused audit sets; Amazon has no audit',
        timing='One complete call logged for accounting only; not compared with original three-repeat latencies',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scientific_base='59d86dbfaaeebb7bdaace151afcec047f3e1a360',
        environment=subprocess.check_output(['uname','-a'],text=True))
    save_json(OUT/'protocol.json',protocol)
    total=0; replay_max=0.
    for name in ['isolet','cifar10','cifar10_gist512','amazon']:
        for phase in (['test','audit'] if name!='amazon' else ['test']):
            data=load_data(name,phase); ev=SprintEvaluator(data)
            queries=data['arrays'][phase]; assert len(queries)==200
            truth=[c.exact_log_mean(ev.points,q,ev.h) for q in queries]
            previous=pd.read_csv(PRIOR/'results/formal'/phase/name/'basic_queries.csv')
            previous=previous[previous.method.isin(['final','matched_single']) & previous.budget.isin(BUDGETS)].set_index(['method','query_id','budget','seed'])
            for ps in SEEDS:
                sp.SEED=ps
                ev.projection=sp.ProjectionScore(data['arrays']['reference'],data['arrays']['train'],ROOT/'models'/('seed_'+str(ps))/name)
                rows=[]
                for qi,q in enumerate(queries):
                    for m in BUDGETS:
                        expected_h=None
                        for method,control in [('final',None),('matched_single','single')]:
                            for seed in range(5):
                                z=ev.evaluate(q,m,np.random.default_rng(seed+qi*100003),CFG,control=control,keep_ids=True)
                                assert z['actual_samples']==z['kernel_evaluations']==m
                                assert len(z['selected_ids'])==len(set(z['selected_ids']))==m
                                assert z['H_count']==m//4 and z['candidate_size']==ev.n and z['residual_size']==0
                                if expected_h is None: expected_h=z['H_ids']
                                assert expected_h==z['H_ids']
                                z.update(c.errors(z.pop('log_estimate'),truth[qi]))
                                if ps==SEEDS[0]:
                                    old=previous.loc[(method,qi,m,seed)]
                                    diff=abs(z['relative_error']-old.relative_error)
                                    replay_max=max(replay_max,diff); assert diff<1e-12,(name,phase,qi,m,seed,diff)
                                z.pop('candidate_ids'); z.pop('selected_ids')
                                z['H_ids']=json.dumps(z['H_ids'])
                                rows.append(dict(dataset=name,phase=phase,projection_seed=ps,method=method,query_id=qi,raw_row_id=int(data['split'][phase][qi]),budget=m,seed=seed,**z))
                pd.DataFrame(rows).to_csv(OUT/f'{name}_{phase}_{ps}.csv',index=False)
                total+=len(rows)
                print(json.dumps(dict(dataset=name,phase=phase,projection_seed=ps,rows=len(rows),total=total,replay_max_error_difference=replay_max)),flush=True)
    save_json(OUT/'PASS.json',dict(rows=total,original_seed_max_error_difference=replay_max,all_budgets_and_H_verified=True,completed_unix=time.time()))

if __name__=='__main__': main()
