"""Diagnostic phase timing, separate from production latency comparisons."""
import time
import numpy as np
from scipy.spatial.distance import cdist
from sprint_estimator import estimate_strata
from upgrade_partition import profile_original


def profile_current(current,q,budget,rng):
    begin=time.perf_counter_ns()
    score=current.projection.query(q)
    H,cells,stages=profile_original(score['d2hat'],current.id_array,budget//4,8)
    stages.update(projection_ns=score['projection_ns'],low_dim_score_ns=score['low_dim_score_ns'])
    detail={}
    ids_seen=[]
    def kernel(ids):
        t=time.perf_counter_ns()
        ids_seen.extend(np.asarray(ids,dtype=np.int64).tolist())
        detail['kernel_id_accounting_ns']=time.perf_counter_ns()-t
        t=time.perf_counter_ns()
        selected=current.points[ids]
        detail['selected_original_vector_copy_ns']=time.perf_counter_ns()-t
        detail['selected_original_vector_copy_count']=len(ids)
        t=time.perf_counter_ns()
        d2=cdist(np.asarray(q,dtype=np.float64)[None,:],selected,'sqeuclidean')[0]
        detail['original_distance_ns']=time.perf_counter_ns()-t
        t=time.perf_counter_ns()
        logs=-d2/(2*current.h*current.h)
        detail['kernel_log_ns']=time.perf_counter_ns()-t
        return logs
    estimate=estimate_strata(current.n,H,cells,budget,rng,kernel,allocation='proportional')
    stages.update({k:int(estimate[k]) for k in ['sampling_ns','allocation_ns','kernel_ns','aggregation_ns']})
    total=time.perf_counter_ns()-begin
    return dict(profile_total_ns=total,log_estimate=estimate['log_estimate'],
        actual_kernel_evaluations=len(ids_seen),**stages,**detail,
        scope='Instrumented diagnostic; nested phase times must not be added twice')
