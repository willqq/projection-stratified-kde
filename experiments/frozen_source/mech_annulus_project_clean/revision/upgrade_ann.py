"""Controlled ANN-H estimators; neither control is named official DEANN.

Official DEANN remains an independent baseline. These controls use uniform
without-replacement draws from the exact complement of the same unique ANN H.
"""
import time
import numpy as np
from scipy.spatial.distance import cdist
from sprint_estimator import estimate_strata, allocate_counts
from upgrade_partition import split_scores


def unique_neighbors(distances, ids, n):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    if ids.shape != distances.shape:
        raise ValueError('Mismatched ANN output shapes')
    valid = (ids >= 0) & (ids < n)
    if np.any(~np.isfinite(distances[valid])) or np.any(distances[valid] < 0):
        raise ValueError('Valid ANN IDs must have finite nonnegative distances')
    H, first = np.unique(ids[valid], return_index=True)
    return H, distances[valid][first], dict(
        ann_valid_returned=int(valid.sum()), ann_invalid_returned=int((~valid).sum()),
        ann_duplicate_returned=int(valid.sum()-len(H)))


class ANNControlledEvaluator:
    def __init__(self, points, h, projection, ann):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.n = len(points)
        self.h = float(h)
        self.projection = projection
        self.ann = ann
        self.ids = np.arange(self.n, dtype=np.int64)
        self.last_partition = None

    def prepare(self, q, budget, fraction, nprobe, mode):
        if mode not in ('single', 'multi'):
            raise ValueError(mode)
        start = time.perf_counter_ns()
        # Charge query-time changes of the shared ANN object's arguments.
        self.ann.set_query_arguments(nprobe)
        requested = min(self.n, max(1, int(budget*fraction)))
        distances, ids, counter = self.ann.query(np.asarray(q,dtype=np.float64), requested)
        ann_ns = time.perf_counter_ns()-start
        ann_stats = dict(self.ann.last)
        t = time.perf_counter_ns()
        H, exact_distances, counters = unique_neighbors(distances, ids, self.n)
        if len(H) >= min(budget,self.n) and len(H)<self.n:
            raise ValueError('ANN H leaves no residual sampling budget')
        dedup_ns = time.perf_counter_ns()-t
        t = time.perf_counter_ns()
        mask = np.ones(self.n,dtype=bool)
        mask[H] = False
        remainder = self.ids[mask]
        complement_ns = time.perf_counter_ns()-t
        phases = dict(ann_retrieval_ns=ann_ns, ann_dedup_ns=dedup_ns,
                      complement_build_ns=complement_ns,
                      projection_ns=0, low_dim_score_ns=0,
                      complement_score_gather_ns=0, sorting_and_strata_ns=0)
        detail={}
        if mode=='multi' and len(remainder):
            if self.projection is None:
                raise RuntimeError('Build fixed projection offline')
            z=self.projection.query(q)
            phases['projection_ns']=z['projection_ns']
            phases['low_dim_score_ns']=z['low_dim_score_ns']
            t=time.perf_counter_ns()
            scores=z['d2hat'][remainder]
            phases['complement_score_gather_ns']=time.perf_counter_ns()-t
            _,cells,detail=split_scores(scores,remainder,0,8)
            phases['sorting_and_strata_ns']=detail['partition_total_ns']
        else:
            cells=[remainder] if len(remainder) else []
        counters.update(ann_distance_evaluations=int(ann_stats['ann_distance_evaluations']),
            ann_recomputed_original_distances=int(ann_stats['neighbors_returned']),
            ann_reported_counter=int(np.asarray(counter).reshape(-1)[0]),
            requested_neighbors=requested, actual_H_count=len(H),
            low_dim_score_evaluations=self.n if mode=='multi' else 0,
            full_dimension_projection_inner_products=64 if mode=='multi' else 0,
            complement_id_scan_count=self.n, **detail)
        return H,cells,exact_distances,phases,counters

    def evaluate(self,q,budget,rng,fraction,nprobe,mode,keep_ids=False):
        start=time.perf_counter_ns()
        H,cells,dH,phases,counters=self.prepare(q,budget,fraction,nprobe,mode)
        called=[]
        original_remainder_evaluations=0
        def kernel_fn(ids):
            nonlocal original_remainder_evaluations
            if called or not np.array_equal(ids[:len(H)],H):
                raise RuntimeError('Expected one proportional call with exact H first')
            called.append(len(ids))
            remainder_ids=ids[len(H):]
            original_remainder_evaluations=len(remainder_ids)
            # ANN wrapper already paid for d-dimensional distances to H. Reuse
            # only those per-query distances; no offline truth enters this path.
            d2=cdist(np.asarray(q,dtype=np.float64)[None,:],
                     self.points[remainder_ids],metric='sqeuclidean')[0]
            return -np.r_[dH*dH,d2]/(2*self.h*self.h)
        estimate=estimate_strata(self.n,H,cells,budget,rng,kernel_fn,
                                 allocation='proportional')
        phases.update({key:int(estimate[key]) for key in
            ['sampling_ns','allocation_ns','kernel_ns','aggregation_ns']})
        total=time.perf_counter_ns()-start
        if sum(called)!=estimate['actual_samples'] or sum(called)>budget:
            raise RuntimeError('Kernel budget mismatch')
        distances=(counters['ann_distance_evaluations']+
                   counters['ann_recomputed_original_distances']+
                   original_remainder_evaluations)
        result=dict(log_estimate=estimate['log_estimate'],time_ns=total,
            actual_samples=estimate['actual_samples'],kernel_evaluations=sum(called),
            full_dimension_query_reference_distances=distances,
            full_dimension_distance_or_dot_evaluations=distances+
                counters['full_dimension_projection_inner_products'],
            H_count=len(H),candidate_size=len(H),residual_size=self.n-len(H),
            original_remainder_distance_evaluations=original_remainder_evaluations,
            layer_sizes=estimate['cell_sizes'].tolist(),allocation=estimate['main_counts'].tolist(),
            effective_allocation=estimate['effective_allocation'],
            nonempty_layers=len(cells),**phases,**counters)
        result['unassigned_ns']=total-sum(phases.values())
        if result['unassigned_ns']<0:
            raise RuntimeError('Overlapping phases')
        self.last_partition=(H,cells,estimate['main_counts'])
        if keep_ids:
            result.update(H_ids=H.tolist(),selected_ids=estimate['selected_ids'].tolist())
        return result


def conditional_variance(values,n,cells,budget):
    """Offline diagnostic only; never called by either online estimator."""
    counts=allocate_counts([len(x) for x in cells],budget)
    return sum(len(c)**2/m*(1-m/len(c))*float(np.var(values[c],ddof=1))
        for c,m in zip(cells,counts) if len(c)>1)/n**2
