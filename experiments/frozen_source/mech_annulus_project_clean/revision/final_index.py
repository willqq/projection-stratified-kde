"""TYPE-1 posting index. Model, thresholds, candidate rule and estimator unchanged."""
import math,time
import numpy as np
import torch
import core as c
from _posting_index import PostingIndex

class IndexedMECH(c.source.MECHHashIndex):
    def __init__(self,model,x,tables,bits,hamming_probe,delta):
        start=time.perf_counter();self.name='MECH-postings';self.model=model.eval();self.x=x
        self.tables_count=tables;self.bits=bits;self.hamming_probe=hamming_probe
        self.thresholds=[];self.point_codes=[]
        with torch.no_grad():hs=self.model.encode_continuous(torch.from_numpy(x))
        for l in range(tables):
            h=hs[l][:,:bits].numpy();t=np.median(h,axis=0);self.thresholds.append(t)
            self.point_codes.append(c.source.pack_bits(h>=t))
        norms=np.linalg.norm(np.asarray(x,dtype=np.float32),axis=1);self.zero_norm_ids=set(np.flatnonzero(norms==0).tolist())
        self.layer_ids=np.floor(norms/delta).astype(int)
        self.postings=PostingIndex([x.tolist() for x in self.point_codes],self.layer_ids.tolist(),sorted(self.zero_norm_ids),bits)
        self.build_time_s=time.perf_counter()-start

class IndexedEvaluator(c.Evaluator):
    def __init__(self,*args,threshold_mapping=None,**kwargs):
        super().__init__(*args,**kwargs);self.threshold_mapping=threshold_mapping
    def thresholds_for_query(self,q):
        a=float(np.linalg.norm(np.asarray(q,dtype=np.float32)))
        angles=[c.source.max_angle_from_radius(a,float(x)) for x in np.linspace(0,self.radius,self.layers+1)[1:]]
        if self.threshold_mapping is None:
            return [max(self.index.hamming_probe,c.source.hamming_radius_from_angle(t,self.index.bits)) for t in angles]
        knots=np.asarray(self.threshold_mapping['angles']);values=np.asarray(self.threshold_mapping['thresholds'])
        # Piecewise-constant upper endpoint bins; monotonic fitted thresholds.
        ids=np.minimum(np.searchsorted(knots,angles,side='left'),len(knots)-1)
        return values[ids].astype(int).tolist()
    def partition(self,q,codes,radius):
        a=float(np.linalg.norm(np.asarray(q,dtype=np.float32)))
        radii=np.linspace(0.,radius,self.layers+1)[1:]
        z=self.index.postings.query(codes,a,radii.tolist(),self.delta,self.min_collisions,self.thresholds_for_query(q))
        self.index.query_stats=z['stats']
        return set(z['union']),{'outer_prefilter_size':z['outer_norm_pool_size']},z['rings']
    def evaluate(self,method,q,budget,rng):
        z=super().evaluate(method,q,budget,rng)
        z['query_encode_ns']=z['encoding_ns'];z['sampling_ns']=z['allocation_ns']
        for k in ['norm_lookup_ns','hash_lookup_ns','candidate_collect_ns','candidate_intersection_ns','dedup_ns','annulus_build_ns']:
            z.setdefault(k,0)
        for k in ['hash_bucket_visits','retrieved_posting_ids','norm_posting_ids','code_trie_node_visits']:z.setdefault(k,0)
        if method=='exact_annular':z['annulus_build_ns']=z['candidate_ns']
        accounted=sum(z[k] for k in ['encoding_ns','radius_ns','norm_lookup_ns','hash_lookup_ns','candidate_collect_ns','candidate_intersection_ns','dedup_ns','annulus_build_ns','sampling_ns','kernel_ns'])
        z['unassigned_ns']=z['time_ns']-accounted
        if z['unassigned_ns']<0:raise RuntimeError('Overlapping timing stages')
        return z
