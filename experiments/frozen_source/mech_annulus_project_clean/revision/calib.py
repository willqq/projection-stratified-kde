"""Empirical monotone calibration of the Hamming radius schedule T(r).

The binomial model E[D_H]=c*theta/pi badly underestimates learned MECH code
distances at small angles (measured ~0.31c bits at 7.5 degrees versus ~0.04c
predicted), collapsing T(r) to its floor. This module fits, on TRAIN queries
against reference points only, the empirical quantile of per-table Hamming
distance as a monotone nondecreasing function of the pairwise angle, and maps
each annulus bound to a Hamming radius through that function.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import mech_annulus_experiments as source

_POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
_POP16 = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    if len(x) and int(x.max()) < (1 << 16):
        return _POP16[x.astype(np.uint16)].astype(np.int16)
    out = np.zeros(len(x), dtype=np.uint8)
    for shift in (0, 8, 16, 24, 32, 40, 48, 56):
        out += _POP[((x >> shift) & 0xFF).astype(np.uint8)]
    return out.astype(np.int16)


def batch_query_codes(index, queries: np.ndarray, batch: int = 1024) -> list[np.ndarray]:
    """Per-table packed codes for many queries, batched through the encoder."""
    per_table: list[list[int]] = [[] for _ in range(index.tables_count)]
    with torch.no_grad():
        for s in range(0, len(queries), batch):
            q = queries[s:s + batch]
            hs = index.model.encode_continuous(torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)))
            for t in range(index.tables_count):
                bits = hs[t][:, :index.bits].numpy() >= index.thresholds[t]
                codes = source.pack_bits(bits)
                per_table[t].extend(int(c) for c in codes)
    return [np.asarray(v, dtype=np.uint64) for v in per_table]


def fit_calibration(index, train_queries: np.ndarray, reference: np.ndarray,
                    quantile: float, seed: int = 20260915, max_pairs: int = 6144,
                    n_bins: int = 18):
    """Return (lookup(theta_rad)->int, bin_mids, bin_radii); monotone in theta.

    Fits the quantile of per-table Hamming distance between sampled
    train-query/reference pairs; a running max enforces monotonicity.
    """
    rng = np.random.default_rng(seed)
    qi = rng.integers(0, len(train_queries), size=max_pairs)
    ri = rng.integers(0, len(reference), size=max_pairs)
    q32 = np.ascontiguousarray(train_queries[qi], dtype=np.float32)
    r32 = reference[ri]
    qn = np.linalg.norm(q32, axis=1).astype(np.float64)
    rn = np.linalg.norm(r32, axis=1).astype(np.float64)
    ok = (qn * rn) > 1e-12
    q32, r32, qn, rn = q32[ok], r32[ok], qn[ok], rn[ok]
    cosang = np.clip((q32.astype(np.float64) * r32.astype(np.float64)).sum(1) / (qn * rn), -1.0, 1.0)
    angles = np.arccos(cosang)
    q_codes = batch_query_codes(index, q32)
    dh_all, ang_rep = [], []
    for t in range(index.tables_count):
        dh_all.append(_popcount(index.point_codes[t][ri] ^ q_codes[t]))
        ang_rep.append(angles)
    dh = np.concatenate(dh_all)
    ang_rep = np.concatenate(ang_rep)
    # Equal-count bins over the observed angle range keep resolution where
    # query/reference angles actually live instead of the full [0, pi].
    order = np.argsort(ang_rep)
    ang_sorted, dh_sorted = ang_rep[order], dh[order]
    quantile_edges = np.quantile(ang_sorted, np.linspace(0.0, 1.0, n_bins + 1))
    quantile_edges[0] -= 1e-9
    quantile_edges[-1] += 1e-9
    T = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = (ang_sorted >= quantile_edges[b]) & (ang_sorted < quantile_edges[b + 1])
        if m.sum() >= 30:
            T[b] = int(np.quantile(dh_sorted[m], quantile))
    T = np.maximum.accumulate(T)  # enforce monotone nondecreasing
    T = np.clip(T, 1, index.bits)
    mids = 0.5 * (quantile_edges[:-1] + quantile_edges[1:])
    bin_edges = quantile_edges

    def lookup(theta: float) -> int:
        theta = min(max(float(theta), 0.0), math.pi)
        b = int(np.searchsorted(bin_edges, theta, side='right') - 1)
        b = min(max(b, 0), n_bins - 1)
        return int(T[b])

    return lookup, mids, T


def calibrated_hamming_radii(q_norm: float, bounds: np.ndarray, lookup) -> list[int]:
    radii = []
    for b in bounds:
        angle = source.max_angle_from_radius(q_norm, float(b))
        radii.append(int(lookup(angle)))
    return radii
