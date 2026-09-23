"""Official DEANN with its official FAISS IVF wrapper and float64 near distances."""
from __future__ import annotations
import math
import sys
import time
import numpy as np
import core as c
import run as r
EXTERNAL=r.JOBROOT/'external/deann'
libs=list((EXTERNAL/'build').rglob('deann*.so'))
if len(libs)!=1: raise ImportError('Expected one built official DEANN extension')
sys.path.insert(0,str(libs[0].parent))
sys.path.insert(0,str(EXTERNAL))
import deann
import faiss
from extern.faiss import FaissIVF


class CommonPrecisionFaiss(FaissIVF):
    def fit(self, X):
        self.reference64=np.ascontiguousarray(X,dtype=np.float64)
        super().fit(X)

    def query(self,q,k):
        t=time.perf_counter_ns()
        _,ids,scanned=super().query(q,k)
        valid=ids>=0
        d=np.full(len(ids),np.inf,dtype=np.float64)
        d[valid]=np.sqrt(c.sqdist(self.reference64[ids[valid]],q))
        self.last={'ann_distance_evaluations':int(scanned[0]),
                   'neighbors_returned':int(valid.sum()),
                   'candidate_ns':time.perf_counter_ns()-t}
        # Official API accepts distances/IDs/sample counters. Count recomputation.
        return d,ids,np.array([int(scanned[0])+int(valid.sum())],dtype=np.int64)


def build(points):
    faiss.omp_set_num_threads(8)
    nlist=min(32,len(points)//40)
    start=time.perf_counter()
    ann=CommonPrecisionFaiss('euclidean',nlist)
    ann.fit(points)
    return ann,{'ann_build_seconds':time.perf_counter()-start,'nlist':nlist,
                'index_bytes':int(faiss.serialize_index(ann.index).nbytes)}


def make_estimator(points,h,ann,budget,fraction,nprobe,seed):
    k=max(1,int(budget*fraction));m=budget-k
    ann.set_query_arguments(nprobe)
    t=time.perf_counter()
    est=deann.AnnEstimatorPermuted(h,'gaussian',k,m,ann,seed)
    est.fit(points)
    return est,k,m,time.perf_counter()-t


def evaluate(est,ann,q,k,m):
    t=time.perf_counter_ns()
    value,samples=est.query(np.ascontiguousarray(q,dtype=np.float64)[None,:])
    total=time.perf_counter_ns()-t
    value=float(value[0]); total_reported=int(samples[0])
    if value<0 or not math.isfinite(value): raise ValueError('Official DEANN returned invalid density')
    nnear=ann.last['neighbors_returned']
    ann_visits=ann.last['ann_distance_evaluations']
    block_samples=total_reported-ann_visits-nnear
    # The permuted implementation computes overlap corrections separately.
    corrections=max(0,block_samples-m)
    return {'log_estimate':math.log(value) if value>0 else -math.inf,
        'time_ns':total,'candidate_ns':ann.last['candidate_ns'],
        'encoding_ns':0,'radius_ns':0,'allocation_ns':0,
        'kernel_ns':total-ann.last['candidate_ns'],
        'actual_samples':nnear+m,'candidate_size':nnear,'nonempty_layers':0,
        'layer_sizes':'[]','allocation':'[]',
        'distance_evaluations':ann_visits+nnear+block_samples+corrections,
        'kernel_evaluations':nnear+block_samples+corrections,
        'hash_id_checks':0,'code_distance_checks':0,
        'ann_distance_evaluations':ann_visits,'ann_neighbors':nnear,
        'requested_neighbors':k,'requested_residual_samples':m,
        'permuted_block_samples':block_samples,'overlap_corrections':corrections,
        'official_samples_counter':total_reported}
