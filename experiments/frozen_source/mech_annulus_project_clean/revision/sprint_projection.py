"""One fixed 64-dimensional Gaussian auxiliary score for the sprint.

This is an O(n*64) scan when candidate_ids is None. It is not a sublinear index.
Only original-space kernels in the estimator define the KDE target.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

SEED = 20260916
PROJECTION_DIMENSION = 64


def _sha_array(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


class ProjectionScore:
    """Frozen train-centered float64 Gaussian projection; no data-based tuning.

    For G[d,64] ~ N(0,1/64), z(x)=(x-train_mean) @ G.
    query returns ||z(p)-z(q)||^2 in candidate_ids order, or reference order.
    Constructor only needs the existing training/reference arrays. It never
    receives validation, test, audit queries, bandwidth, or ground-truth KDE.
    """
    def __init__(self, reference32, train32, cache_dir=None):
        start = time.perf_counter_ns()
        reference = np.asarray(reference32, dtype=np.float32)
        training = np.asarray(train32, dtype=np.float32)
        if reference.ndim != 2 or training.ndim != 2 or reference.shape[1] != training.shape[1]:
            raise ValueError('Reference and training feature dimensions must match')
        if len(reference) == 0 or len(training) == 0 or not np.isfinite(reference).all() or not np.isfinite(training).all():
            raise ValueError('Finite nonempty training and reference required')
        self.n, self.dimension = reference.shape
        self.output_dimension = PROJECTION_DIMENSION
        signature = dict(seed=SEED,projection_dimension=PROJECTION_DIMENSION,
            projection_dtype='float64',reference_sha256=_sha_array(reference),
            training_sha256=_sha_array(training),reference_shape=list(reference.shape),
            training_shape=list(training.shape),distribution='N(0, 1/64) iid',
            centering='float64 training arithmetic mean')
        folder = Path(cache_dir) if cache_dir is not None else None
        path = folder/'projection64.npz' if folder is not None else None
        self.cache_loaded = bool(path is not None and path.exists())
        if self.cache_loaded:
            record = json.loads((folder/'metadata.json').read_text())
            if record['signature'] != signature:
                raise RuntimeError('Projection cache inputs do not match frozen configuration')
            with np.load(path) as stored:
                self.mean = stored['training_mean']
                self.matrix = stored['gaussian_matrix']
                self.reference_projected = stored['reference_projected']
                self.reference_projected_norm2 = stored['reference_projected_norm2']
            for name,array in [('training_mean',self.mean),('gaussian_matrix',self.matrix),
                               ('reference_projected',self.reference_projected),
                               ('reference_projected_norm2',self.reference_projected_norm2)]:
                if _sha_array(array) != record['array_sha256'][name]:
                    raise RuntimeError('Projection cache checksum mismatch: '+name)
            self.fit_record = record
        else:
            self.mean = np.mean(training,axis=0,dtype=np.float64)
            rng = np.random.default_rng(SEED)
            self.matrix = np.ascontiguousarray(rng.normal(0.,1/np.sqrt(PROJECTION_DIMENSION),
                                            size=(self.dimension,PROJECTION_DIMENSION)),dtype=np.float64)
            self.reference_projected = np.ascontiguousarray(
                (np.asarray(reference,dtype=np.float64)-self.mean) @ self.matrix,dtype=np.float64)
            self.reference_projected_norm2 = np.einsum('ij,ij->i',self.reference_projected,self.reference_projected)
            arrays = dict(training_mean=self.mean,gaussian_matrix=self.matrix,
                          reference_projected=self.reference_projected,
                          reference_projected_norm2=self.reference_projected_norm2)
            self.fit_record = dict(signature=signature,
                array_sha256={name:_sha_array(array) for name,array in arrays.items()},
                build_ns=time.perf_counter_ns()-start,
                offline_full_dimension_projection_inner_products=self.n*PROJECTION_DIMENSION,
                matrix_bytes=int(self.matrix.nbytes),training_mean_bytes=int(self.mean.nbytes),
                reference_projection_bytes=int(self.reference_projected.nbytes),
                reference_projected_norm_bytes=int(self.reference_projected_norm2.nbytes),
                auxiliary_reference_bytes=int(self.reference_projected.nbytes+self.reference_projected_norm2.nbytes))
            if folder is not None:
                folder.mkdir(parents=True,exist_ok=True)
                np.savez(path,**arrays)
                (folder/'metadata.json').write_text(json.dumps(self.fit_record,indent=2))
        self.offline_record = dict(self.fit_record,cache_loaded=self.cache_loaded,
            load_or_build_ns=time.perf_counter_ns()-start,
            storage_bytes=int(self.mean.nbytes+self.matrix.nbytes+self.reference_projected.nbytes+self.reference_projected_norm2.nbytes),
            query_scan_complexity='O(|C|*64); C=P when candidate_ids=None',
            target_kernel_changed=False)

    def query(self, q32, candidate_ids=None):
        start = time.perf_counter_ns()
        q = np.asarray(q32,dtype=np.float64)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError('Finite query with original feature dimension required')
        t = time.perf_counter_ns()
        projected = (q-self.mean) @ self.matrix
        projection_ns = time.perf_counter_ns()-t
        t = time.perf_counter_ns()
        if candidate_ids is None:
            refs = self.reference_projected
            norm2 = self.reference_projected_norm2
            count = self.n
        else:
            ids = np.asarray(candidate_ids,dtype=np.int64)
            refs = self.reference_projected[ids]
            norm2 = self.reference_projected_norm2[ids]
            count = len(ids)
        d2 = np.maximum(0.,norm2 + float(projected@projected) - 2*(refs@projected))
        score_ns = time.perf_counter_ns()-t
        return dict(d2hat=d2,projection_ns=projection_ns,low_dim_score_ns=score_ns,
            total_ns=time.perf_counter_ns()-start,
            full_dimension_projection_inner_products=PROJECTION_DIMENSION,
            full_dimension_query_reference_distances=0,
            low_dim_score_evaluations=count,low_dim_score_dimension=PROJECTION_DIMENSION,
            reference_projection_bytes=int(self.reference_projected.nbytes),
            reference_projected_norm_bytes=int(self.reference_projected_norm2.nbytes),
            projection_matrix_bytes=int(self.matrix.nbytes),training_mean_bytes=int(self.mean.nbytes),
            offline_build_ns=int(self.fit_record['build_ns']))
