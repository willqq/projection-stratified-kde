"""Online norm-angle score strata and full-population hybrid estimator."""
import math,time
import numpy as np
import core as c
from sprint_common import ROOT
from sprint_scores import ScoreModel,quantile_groups
from sprint_estimator import estimate_strata
from sprint_projection import ProjectionScore

class SprintEvaluator:
    def __init__(self,data):
        self.data=data;self.old=data['old'];self.points=data['points'];self.n=len(self.points);self.h=self.old.h
        self.score=ScoreModel(data['arrays']['reference'],data['arrays']['train'],data['model'],data['index'],ROOT/'models/score_calibration'/data['name'])
        self.id_array=np.arange(self.n,dtype=np.int64)
        self.projection=None
    def ensure_projection(self):
        if self.projection is None:
            self.projection=ProjectionScore(self.data["arrays"]["reference"],self.data["arrays"]["train"],ROOT/"models/projection64"/self.data["name"])
        return self.projection
    def prepare(self,q,budget,cfg,control=None):
        if cfg['score']=='projection64' and cfg['grouping']!='quantile':raise ValueError('Projection route requires quantile grouping')
        if not cfg['full_target'] and (cfg['H_fraction'] != 0 or control is not None):
            raise ValueError('Candidate-only diagnostic supports H=0 and no full-target control')
        start=time.perf_counter_ns();stages={};counts={}
        if cfg['score']=='projection64':
            if self.projection is None:raise RuntimeError('Projection must be built offline before evaluating')
            C=self.id_array;rings=[];z=self.projection.query(q)
            stages['projection_ns']=z['projection_ns'];stages['low_dim_score_ns']=z['low_dim_score_ns']
            counts.update(full_dimension_encoder_inner_products=0,full_dimension_projection_inner_products=64,
              low_dim_score_evaluations=self.n,low_dim_score_dimension=64,hash_bucket_visits=0,retrieved_posting_ids=0,
              code_trie_node_visits=0,norm_posting_ids=0,hamming_table_comparisons=0,full_dimension_query_norm_evaluations=0)
        else:
            t=time.perf_counter_ns();enc=self.score.encode_query(q);stages['query_encode_ns']=time.perf_counter_ns()-t
            counts['full_dimension_encoder_inner_products']=enc['full_dimension_encoder_inner_products']
            t=time.perf_counter_ns();union,_,rings=self.old.partition(q,enc['codes'],self.old.radius)
            C=np.asarray(sorted(union),dtype=np.int64);stages['candidate_retrieval_ns']=time.perf_counter_ns()-t
            rawstats=self.old.index.query_stats.copy()
            for k in ['hash_bucket_visits','retrieved_posting_ids','code_trie_node_visits','norm_posting_ids']:counts[k]=int(rawstats.get(k,0))
            t=time.perf_counter_ns();z=self.score.query(q,C,codes=enc['codes'],encoded=enc,kind=cfg['score']);stages['score_ns']=time.perf_counter_ns()-t
            counts['low_dim_score_evaluations']=len(C);counts['low_dim_score_dimension']=60
            counts['hamming_table_comparisons']=z['hamming_table_comparisons']
            counts['full_dimension_query_norm_evaluations']=3
        t=time.perf_counter_ns();order=np.lexsort((C,z['d2hat']));ranked=C[order]
        k=min(int(budget*cfg['H_fraction']),len(C),max(0,budget-1));H=ranked[:k]
        if control=='single':
            cells=[]
        elif cfg['grouping']=='old':
            cells=[np.setdiff1d(np.asarray(cell,dtype=np.int64),H,assume_unique=False) for cell in rings]
            cells=[cell for cell in cells if len(cell)]
        else:
            cells=[x for x in np.array_split(ranked[k:],min(cfg['J'],max(1,len(C)-k))) if len(x)]
        stages['sorting_and_strata_ns']=time.perf_counter_ns()-t
        t=time.perf_counter_ns()
        if control=='single':
            remainder=ranked[k:] if cfg['score']=='projection64' else np.setdiff1d(self.id_array,H,assume_unique=True);cells=[remainder] if len(remainder) else []
            residual_size=self.n-len(C)
        elif cfg['full_target']:
            residual=np.empty(0,dtype=np.int64) if cfg['score']=='projection64' else np.setdiff1d(self.id_array,C,assume_unique=True)
            residual_size=len(residual)
            if residual_size:cells.append(residual)
        else:residual_size=self.n-len(C)
        stages['residual_build_ns']=time.perf_counter_ns()-t
        counts['residual_id_scan_count']=self.n if ((control=='single' or cfg['full_target']) and cfg['score']!='projection64') else 0
        return dict(H=H,cells=cells,C=C,residual_size=residual_size,stages=stages,counts=counts,prepare_ns=time.perf_counter_ns()-start)
    def evaluate(self,q,budget,rng,cfg,control=None,keep_ids=False):
        start=time.perf_counter_ns();p=self.prepare(q,budget,cfg,control)
        measured_kernel_ids=[]
        def kernel_fn(ids):
            measured_kernel_ids.extend(np.asarray(ids,dtype=np.int64).tolist())
            return -c.sqdist(self.points[ids],q)/(2*self.h*self.h)
        allocation=cfg['allocation']
        if control=='single' or control=='multi_proportional':allocation='proportional'
        if control=='multi_neyman':allocation='neyman'
        if cfg['full_target'] or control is not None:
            est=estimate_strata(self.n,p['H'],p['cells'],budget,rng,kernel_fn,allocation=allocation,pilot_max=2,shrinkage=.5)
        else:
            t=time.perf_counter_ns();cells=c.prepare_rings(p['cells'],budget);alloc=c.allocate([len(x) for x in cells],budget);ids,w=c.sample_ids(cells,alloc,rng);sampling_ns=time.perf_counter_ns()-t
            t=time.perf_counter_ns();logs=kernel_fn(ids) if len(ids) else np.zeros(0);kernel_ns=time.perf_counter_ns()-t
            t=time.perf_counter_ns();value=c.log_weighted_estimate(logs,w,self.n);aggregation_ns=time.perf_counter_ns()-t
            est=dict(log_estimate=value,actual_samples=len(ids),selected_ids=ids,H_count=0,cell_sizes=np.array([len(x) for x in cells]),pilot_counts=np.zeros(len(cells),dtype=int),main_counts=alloc,effective_allocation='proportional',fallback='',sampling_ns=sampling_ns,allocation_ns=0,kernel_ns=kernel_ns,aggregation_ns=aggregation_ns,kernel_calls=int(len(ids)>0),kernel_call_sizes=[len(ids)] if len(ids) else [])
        phases=dict(p['stages'],**{k:int(est[k]) for k in ['sampling_ns','allocation_ns','kernel_ns','aggregation_ns']})
        total=time.perf_counter_ns()-start
        actual=len(measured_kernel_ids)
        if actual!=est['actual_samples'] or actual>budget:raise RuntimeError('Budget accounting mismatch')
        row=dict(log_estimate=est['log_estimate'],time_ns=total,actual_samples=actual,kernel_evaluations=actual,full_dimension_query_reference_distances=actual,
          full_dimension_distance_or_dot_evaluations=actual+p['counts']['full_dimension_encoder_inner_products']+p['counts'].get('full_dimension_projection_inner_products',0),candidate_size=len(p['C']),residual_size=p['residual_size'],H_count=est['H_count'],
          nonempty_layers=len(est['cell_sizes']),layer_sizes=est['cell_sizes'].tolist(),pilot_counts=est['pilot_counts'].tolist(),allocation=est['main_counts'].tolist(),effective_allocation=est['effective_allocation'],fallback=est['fallback'],kernel_calls=est['kernel_calls'],kernel_call_sizes=est['kernel_call_sizes'],
          **phases,**p['counts'])
        row['unassigned_ns']=total-sum(phases.values())
        if row['unassigned_ns']<0:raise RuntimeError('Overlapping query phases')
        if keep_ids:row.update(selected_ids=est['selected_ids'].tolist(),H_ids=p['H'].tolist(),candidate_ids=p['C'].tolist())
        return row
