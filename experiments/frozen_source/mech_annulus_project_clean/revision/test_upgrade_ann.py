"""Independent finite-population checks for controlled ANN-H estimators."""
import json
import numpy as np
from scipy.spatial.distance import cdist
from upgrade_ann import ANNControlledEvaluator, unique_neighbors, conditional_variance
from sprint_projection import ProjectionScore


class FixedANN:
    def __init__(self,points,ids):self.points=points;self.ids=np.array(ids,dtype=np.int64)
    def set_query_arguments(self,nprobe):self.nprobe=nprobe
    def query(self,q,k):
        ids=self.ids[:k];valid=ids>=0
        d=np.full(len(ids),np.inf)
        d[valid]=np.sqrt(cdist(q[None,:],self.points[ids[valid]],'sqeuclidean')[0])
        self.last=dict(ann_distance_evaluations=17,neighbors_returned=int(valid.sum()))
        return d,ids,np.array([17+valid.sum()])


def run():
    rng=np.random.default_rng(20260920)
    points=rng.normal(size=(39,7)).astype(np.float32)
    q=rng.normal(size=7).astype(np.float32);h=2.
    projection=ProjectionScore(points,points[:20])
    ann=FixedANN(points,[3,7,7,-1,18,23,2,13])
    ev=ANNControlledEvaluator(points,h,projection,ann)
    values=np.exp(-cdist(q.astype(float)[None,:],points.astype(float),'sqeuclidean')[0]/(2*h*h))
    truth=values.mean();report={}
    for mode in ['single','multi']:
        estimates=[]
        for seed in range(4096):
            row=ev.evaluate(q,24,np.random.default_rng(seed),.25,2,mode,keep_ids=True)
            H,cells,alloc=ev.last_partition
            combined=np.concatenate([H]+cells)
            assert sorted(combined)==list(range(len(points)))
            assert row['H_ids']==[3,7,18,23]
            assert row['ann_duplicate_returned']==1 and row['ann_invalid_returned']==1
            assert row['actual_samples']==row['kernel_evaluations']==24
            assert len(set(row['selected_ids']))==24
            assert row['original_remainder_distance_evaluations']==20
            assert row['full_dimension_query_reference_distances']==17+5+20
            estimates.append(np.exp(row['log_estimate']))
        var=conditional_variance(values,len(points),cells,24-len(H))
        error=abs(np.mean(estimates)-truth)
        assert error<5*np.sqrt(var/len(estimates))
        report[mode]=dict(mean=float(np.mean(estimates)),truth=float(truth),
            analytical_variance=float(var),empirical_variance=float(np.var(estimates)),
            absolute_mean_error=float(error),repetitions=len(estimates),actual_H=len(H),
            actual_kernel_evaluations=24)
    report['scope']='Synthetic correctness; ANN outputs intentionally include duplicates and missing IDs'
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
