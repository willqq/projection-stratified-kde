"""Frozen-score adapters for the 2026-09-16 sprint.

Only train/reference pairs fit score-to-angle maps. Query receives no true
query/reference distances. Equal-occupancy groups use reference ID to break ties.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

SEED = 20260916
PAIR_COUNT = 4096


def quantile_groups(candidate_ids, approximate_d2, layers):
    ids = np.asarray(candidate_ids, dtype=np.int64)
    d2 = np.asarray(approximate_d2, dtype=np.float64)
    if len(ids) != len(d2) or layers < 1:
        raise ValueError('Invalid grouping input')
    if not np.isfinite(d2).all() or len(np.unique(ids)) != len(ids):
        raise ValueError('Finite scores and distinct candidate IDs required')
    order = np.lexsort((ids, d2))
    return [x for x in np.array_split(ids[order], min(layers, len(ids))) if len(x)] if len(ids) else []


def _normed_latents(model, x, tables, bits):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            hs = model.encode_continuous(torch.from_numpy(np.asarray(x[i:i+256], dtype=np.float32)))
            z = np.stack([h[:, :bits].numpy() for h in hs[:tables]], axis=1)
            lengths = np.linalg.norm(z, axis=2, keepdims=True)
            z = z / np.maximum(lengths, np.finfo(np.float32).tiny)
            out.append(z.reshape(len(z), tables * bits) / np.sqrt(tables))
    return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)


class ScoreModel:
    """Low-dimensional score and train-only monotone angular calibration.

    The real score averages per-encoder tanh-latent cosine similarities, matching
    the existing model's cosine training objective. No network weights are fit.
    All stored reference norms refer to the original transformed feature space.
    query(..., encoded=encode_query(q)) can reuse one MECH forward for retrieval
    and real scoring. Passing just codes avoids a second forward for Hamming.
    """
    def __init__(self, reference32, train32, model, index, cache_dir=None):
        start = time.perf_counter_ns()
        self.model = model.eval()
        self.index = index
        self.tables = int(index.tables_count)
        self.bits = int(index.bits)
        self.dimension = reference32.shape[1]
        self.n = len(reference32)
        self.reference_norms = np.linalg.norm(np.asarray(reference32, dtype=np.float64), axis=1)
        self.reference_codes = np.ascontiguousarray(np.stack(index.point_codes, axis=1), dtype=np.int64)
        self.reference_real = _normed_latents(model, reference32, self.tables, self.bits)
        self.popcount = np.asarray([bin(i).count('1') for i in range(1 << self.bits)], dtype=np.uint8)
        self.maps = {}
        folder = Path(cache_dir) if cache_dir is not None else None
        marker = folder / 'calibration.npz' if folder is not None else None
        if marker is not None and marker.exists():
            z = np.load(marker)
            for kind in ('hamming', 'real'):
                self.maps[kind] = (z[kind+'_x'], z[kind+'_theta'])
            self.fit_record = json.loads((folder/'metadata.json').read_text())
        else:
            self._fit(reference32, train32)
            if folder is not None:
                folder.mkdir(parents=True, exist_ok=True)
                arrays = {}
                for kind, (x, y) in self.maps.items():
                    arrays[kind+'_x'], arrays[kind+'_theta'] = x, y
                arrays.update(training_pair_ids=self.training_pair_ids,
                              reference_pair_ids=self.reference_pair_ids)
                np.savez(marker, **arrays)
                (folder/'metadata.json').write_text(json.dumps(self.fit_record, indent=2))
        self.offline_record = dict(self.fit_record,
            load_or_build_ns=time.perf_counter_ns()-start,
            extra_reference_real_bytes=int(self.reference_real.nbytes),
            extra_reference_norm_bytes=int(self.reference_norms.nbytes),
            auxiliary_reference_code_bytes=int(self.reference_codes.nbytes),
            auxiliary_popcount_bytes=int(self.popcount.nbytes))

    def _fit(self, reference32, train32):
        t = time.perf_counter_ns()
        rng = np.random.default_rng(SEED)
        it = rng.integers(len(train32), size=PAIR_COUNT)
        ir = rng.integers(len(reference32), size=PAIR_COUNT)
        self.training_pair_ids, self.reference_pair_ids = it, ir
        train_real = _normed_latents(self.model, train32, self.tables, self.bits)
        train_codes = []
        with torch.no_grad():
            for i in range(0, len(train32), 256):
                hs = self.model.encode_continuous(torch.from_numpy(train32[i:i+256]))
                code = []
                for l in range(self.tables):
                    binary = hs[l][:, :self.bits].numpy() >= self.index.thresholds[l]
                    code.append(np.sum(binary.astype(np.int64) * (1 << np.arange(self.bits)), axis=1))
                train_codes.append(np.stack(code, axis=1))
        train_codes = np.concatenate(train_codes)
        true_theta = []
        for j in range(0, PAIR_COUNT, 128):
            a = np.asarray(train32[it[j:j+128]], dtype=np.float64)
            b = np.asarray(reference32[ir[j:j+128]], dtype=np.float64)
            prod = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            cosine = np.divide(np.einsum('ij,ij->i', a, b), prod,
                               out=np.ones(len(a)), where=prod > 0)
            true_theta.extend(np.arccos(np.clip(cosine, -1, 1)))
        theta = np.asarray(true_theta)
        hamming = self.popcount[np.bitwise_xor(train_codes[it], self.reference_codes[ir])].sum(axis=1)/(self.tables*self.bits)
        real = 1-np.einsum('ij,ij->i', train_real[it].astype(np.float64), self.reference_real[ir].astype(np.float64))
        details = {}
        for kind, score in [('hamming', hamming), ('real', real)]:
            iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(score, theta)
            self.maps[kind] = (iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float))
            details[kind] = dict(unique_training_scores=int(len(np.unique(score))),
                                knots=int(len(iso.X_thresholds_)),
                                training_angle_mae=float(np.mean(np.abs(iso.predict(score)-theta))))
        self.fit_record = dict(seed=SEED, pair_count=PAIR_COUNT, calibration_source='train/reference only',
            model_retrained=False, real_score='1 - mean per-table tanh-latent cosine',
            monotone_direction='increasing', table_count=self.tables, bits=self.bits,
            fit_ns=time.perf_counter_ns()-t, score_fits=details,
            tie_rule='lexicographic (approximate squared distance, reference ID)')

    def encode_query(self, q32):
        t = time.perf_counter_ns()
        with torch.no_grad():
            hs = self.model.encode_continuous(torch.from_numpy(np.asarray(q32, dtype=np.float32))[None, :])
        codes = []
        z = []
        for l in range(self.tables):
            h = hs[l][0, :self.bits].numpy()
            codes.append(int(np.sum((h >= self.index.thresholds[l]).astype(np.int64) * (1 << np.arange(self.bits)))))
            z.append(h / max(float(np.linalg.norm(h)), np.finfo(np.float32).tiny))
        real = np.ascontiguousarray(np.concatenate(z) / np.sqrt(self.tables), dtype=np.float32)
        return dict(codes=codes, real=real, query_encode_ns=time.perf_counter_ns()-t,
                    full_dimension_encoder_inner_products=self.tables*self.bits)

    def query(self, q32, candidate_ids, codes=None, kind='hamming', encoded=None):
        start = time.perf_counter_ns()
        ids = np.asarray(candidate_ids, dtype=np.int64)
        if kind not in ('hamming', 'real'):
            raise ValueError(kind)
        encoding_ns = 0
        encoder_dots = 0
        if encoded is None and (codes is None or kind == 'real'):
            encoded = self.encode_query(q32)
            encoding_ns = encoded['query_encode_ns']
            encoder_dots = encoded['full_dimension_encoder_inner_products']
        if codes is None:
            codes = encoded['codes']
        t = time.perf_counter_ns()
        if kind == 'hamming':
            raw = self.popcount[np.bitwise_xor(self.reference_codes[ids], np.asarray(codes))].sum(axis=1)/(self.tables*self.bits)
        else:
            raw = 1-np.asarray(self.reference_real[ids], dtype=np.float64) @ np.asarray(encoded['real'], dtype=np.float64)
        score_ns = time.perf_counter_ns()-t
        t = time.perf_counter_ns()
        x, theta = self.maps[kind]
        angle = np.interp(raw, x, theta)
        calibration_ns = time.perf_counter_ns()-t
        t = time.perf_counter_ns()
        qnorm = float(np.linalg.norm(np.asarray(q32, dtype=np.float64)))
        pnorm = self.reference_norms[ids]
        d2 = np.maximum(0., qnorm*qnorm + pnorm*pnorm - 2*qnorm*pnorm*np.cos(angle))
        norm_combine_ns = time.perf_counter_ns()-t
        return dict(d2hat=d2, raw_score=raw, theta_hat=angle, query_norm=qnorm,
            query_encode_ns=encoding_ns, low_dim_score_ns=score_ns,
            score_calibration_ns=calibration_ns, norm_combine_ns=norm_combine_ns,
            score_total_ns=time.perf_counter_ns()-start,
            low_dim_score_evaluations=len(ids), low_dim_score_dimension=self.tables*self.bits,
            hamming_table_comparisons=len(ids)*self.tables if kind=='hamming' else 0,
            original_space_query_norm_evaluations=1,
            full_dimension_encoder_inner_products=encoder_dots,
            full_dimension_query_reference_distances=0)
