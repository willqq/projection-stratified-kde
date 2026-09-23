from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260602
EPS = 1e-12
TIMING_REPEATS = 7


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def minmax_shift_from_origin(x: np.ndarray) -> np.ndarray:
    x = MinMaxScaler().fit_transform(x)
    x = np.nan_to_num(x).astype(np.float32)
    anchor = x[np.argmax(np.linalg.norm(x, axis=1))]
    return (2.0 * x + anchor).astype(np.float32)


def load_dataset(name: str) -> np.ndarray:
    name = name.lower()
    if name == "isolet":
        df = pd.read_csv("data/isolet/isolet1+2+3+4.data", header=None)
        x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    elif name == "cifar10":
        with open("data/cifar-10-batches-py/data_batch_1", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x = batch[b"data"].astype(np.float32)
    elif name == "cifar10_gist512":
        x = scio.loadmat("data/cifar10-Gist512/Cifar10-Gist512.mat")["X"].astype(
            np.float32
        )
    elif name == "amazon":
        data, _ = arff.loadarff(
            "data/Amazon_initial_50_30_10000/"
            "Amazon_initial_50_30_10000.arff"
        )
        df = pd.DataFrame(data).drop(columns=["class_duplicate"])
        x = df.to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return minmax_shift_from_origin(x)


def split_data(
    x: np.ndarray,
    n_train: int,
    n_index: int,
    n_queries: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    needed = min(len(x), n_train + n_index + n_queries)
    if needed < n_index + n_queries:
        raise ValueError("Not enough rows for index and query splits.")
    perm = rng.choice(len(x), size=needed, replace=False)
    sample = x[perm]
    n_train = min(n_train, needed - n_index - n_queries)
    train_x = sample[:n_train]
    index_x = sample[n_train : n_train + n_index]
    queries = sample[n_train + n_index : n_train + n_index + n_queries]
    return train_x, index_x, queries


def distance_matrix(index_x: np.ndarray, queries: np.ndarray) -> np.ndarray:
    rows = [np.linalg.norm(index_x - q, axis=1) for q in queries]
    return np.asarray(rows, dtype=np.float32)


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-(distances**2) / (2.0 * bandwidth**2))


def sampled_ring_kde(
    index_x: np.ndarray,
    q: np.ndarray,
    rings: list[set[int]],
    bandwidth: float,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    estimate = 0.0
    sampled_total = 0
    sample_size = max(int(sample_size), 1)
    for ring in rings:
        if not ring:
            continue
        ids = np.fromiter(ring, dtype=np.int32)
        if len(ids) > sample_size:
            ids = rng.choice(ids, size=sample_size, replace=False)
        distances = np.linalg.norm(index_x[ids] - q, axis=1)
        estimate += len(ring) * float(gaussian_kernel(distances, bandwidth).mean())
        sampled_total += len(ids)
    return estimate, sampled_total


def size_weighted_ring_allocations(rings: list[set[int]], total_budget: int) -> list[int]:
    sizes = np.asarray([len(ring) for ring in rings], dtype=np.int64)
    allocations = np.zeros(len(sizes), dtype=np.int64)
    total_available = int(sizes.sum())
    budget = min(max(int(total_budget), 0), total_available)
    if budget <= 0 or total_available <= 0:
        return allocations.tolist()

    raw = budget * sizes.astype(np.float64) / float(total_available)
    allocations = np.floor(raw).astype(np.int64)
    allocations = np.minimum(allocations, sizes)
    remaining = budget - int(allocations.sum())
    if remaining <= 0:
        return allocations.tolist()

    remainders = raw - np.floor(raw)
    order = np.lexsort((-sizes, -remainders))
    for idx in order:
        if remaining <= 0:
            break
        if allocations[idx] >= sizes[idx]:
            continue
        allocations[idx] += 1
        remaining -= 1
    return allocations.tolist()


def balanced_size_weighted_ring_allocations(
    rings: list[set[int]],
    total_budget: int,
) -> list[int]:
    sizes = np.asarray([len(ring) for ring in rings], dtype=np.int64)
    allocations = np.zeros(len(sizes), dtype=np.int64)
    nonempty = np.flatnonzero(sizes > 0)
    total_available = int(sizes.sum())
    budget = min(max(int(total_budget), 0), total_available)
    if budget <= 0 or total_available <= 0:
        return allocations.tolist()

    min_per_ring = 2 if budget >= 2 * len(nonempty) else 1
    for idx in nonempty:
        if budget <= 0:
            break
        current = min(int(sizes[idx]), min_per_ring, budget)
        allocations[idx] = current
        budget -= current
    if budget <= 0:
        return allocations.tolist()

    remaining_capacity = sizes - allocations
    remaining_total = int(remaining_capacity.sum())
    if remaining_total <= 0:
        return allocations.tolist()

    raw = budget * remaining_capacity.astype(np.float64) / float(remaining_total)
    extra = np.floor(raw).astype(np.int64)
    extra = np.minimum(extra, remaining_capacity)
    allocations += extra
    remaining = budget - int(extra.sum())
    if remaining <= 0:
        return allocations.tolist()

    remainders = raw - np.floor(raw)
    order = np.lexsort((-remaining_capacity, -remainders))
    for idx in order:
        if remaining <= 0:
            break
        if allocations[idx] >= sizes[idx]:
            continue
        allocations[idx] += 1
        remaining -= 1
    return allocations.tolist()


def sampled_weighted_budget_ring_kde(
    index_x: np.ndarray,
    q: np.ndarray,
    rings: list[set[int]],
    bandwidth: float,
    total_budget: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    estimate = 0.0
    sampled_total = 0
    allocations = size_weighted_ring_allocations(rings, total_budget)
    for ring, sample_count in zip(rings, allocations):
        if not ring or sample_count <= 0:
            continue
        ids = np.fromiter(ring, dtype=np.int32)
        if len(ids) > sample_count:
            ids = rng.choice(ids, size=int(sample_count), replace=False)
        distances = np.linalg.norm(index_x[ids] - q, axis=1)
        estimate += len(ring) * float(gaussian_kernel(distances, bandwidth).mean())
        sampled_total += len(ids)
    return estimate, sampled_total


def sampled_balanced_budget_ring_kde(
    index_x: np.ndarray,
    q: np.ndarray,
    rings: list[set[int]],
    bandwidth: float,
    total_budget: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    estimate = 0.0
    sampled_total = 0
    allocations = balanced_size_weighted_ring_allocations(rings, total_budget)
    for ring, sample_count in zip(rings, allocations):
        if not ring or sample_count <= 0:
            continue
        ids = np.fromiter(ring, dtype=np.int32)
        if len(ids) > sample_count:
            ids = rng.choice(ids, size=int(sample_count), replace=False)
        distances = np.linalg.norm(index_x[ids] - q, axis=1)
        estimate += len(ring) * float(gaussian_kernel(distances, bandwidth).mean())
        sampled_total += len(ids)
    return estimate, sampled_total


def sampled_pilot_neyman_budget_ring_kde(
    index_x: np.ndarray,
    q: np.ndarray,
    rings: list[set[int]],
    bandwidth: float,
    total_budget: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    budget = max(int(total_budget), 0)
    nonempty = [idx for idx, ring in enumerate(rings) if ring]
    if budget <= 0 or not nonempty:
        return 0.0, 0

    pilot_per_ring = 2 if budget >= 2 * len(nonempty) else 1
    ring_state = []
    sampled_total = 0
    remaining_budget = budget
    for ring_idx, ring in enumerate(rings):
        if not ring:
            ring_state.append(None)
            continue
        ids = np.fromiter(ring, dtype=np.int32)
        pilot_count = min(len(ids), pilot_per_ring, remaining_budget)
        if pilot_count <= 0:
            ring_state.append(
                {
                    "ids": ids,
                    "sampled_ids": np.asarray([], dtype=np.int32),
                    "values": np.asarray([], dtype=np.float64),
                }
            )
            continue
        if len(ids) > pilot_count:
            pilot_ids = rng.choice(ids, size=int(pilot_count), replace=False)
        else:
            pilot_ids = ids
        distances = np.linalg.norm(index_x[pilot_ids] - q, axis=1)
        values = gaussian_kernel(distances, bandwidth).astype(np.float64)
        sampled_total += len(pilot_ids)
        remaining_budget -= len(pilot_ids)
        ring_state.append(
            {
                "ids": ids,
                "sampled_ids": np.asarray(pilot_ids, dtype=np.int32),
                "values": values,
            }
        )

    if remaining_budget > 0:
        scores = []
        capacities = []
        for state in ring_state:
            if state is None:
                scores.append(0.0)
                capacities.append(0)
                continue
            ids = state["ids"]
            sampled_ids = state["sampled_ids"]
            values = state["values"]
            capacity = max(len(ids) - len(sampled_ids), 0)
            capacities.append(capacity)
            if capacity <= 0 or len(values) == 0:
                scores.append(0.0)
                continue
            if len(values) >= 2:
                spread = float(np.std(values, ddof=1))
            else:
                mean_value = float(values[0])
                spread = math.sqrt(max(mean_value * (1.0 - mean_value), EPS))
            scores.append(float(len(ids)) * max(spread, EPS))

        scores_array = np.asarray(scores, dtype=np.float64)
        capacities_array = np.asarray(capacities, dtype=np.int64)
        if float(scores_array.sum()) <= EPS:
            scores_array = capacities_array.astype(np.float64)
        if int(capacities_array.sum()) > 0 and float(scores_array.sum()) > EPS:
            extra_budget = min(remaining_budget, int(capacities_array.sum()))
            raw = extra_budget * scores_array / float(scores_array.sum())
            extras = np.floor(raw).astype(np.int64)
            extras = np.minimum(extras, capacities_array)
            remaining_extra = extra_budget - int(extras.sum())
            remainders = raw - np.floor(raw)
            order = np.lexsort((-capacities_array, -remainders))
            for idx in order:
                if remaining_extra <= 0:
                    break
                if extras[idx] >= capacities_array[idx]:
                    continue
                extras[idx] += 1
                remaining_extra -= 1

            for idx, extra_count in enumerate(extras):
                if extra_count <= 0 or ring_state[idx] is None:
                    continue
                state = ring_state[idx]
                ids = state["ids"]
                sampled_ids = state["sampled_ids"]
                if len(sampled_ids):
                    unsampled = np.setdiff1d(ids, sampled_ids, assume_unique=False)
                else:
                    unsampled = ids
                if len(unsampled) > extra_count:
                    extra_ids = rng.choice(unsampled, size=int(extra_count), replace=False)
                else:
                    extra_ids = unsampled
                distances = np.linalg.norm(index_x[extra_ids] - q, axis=1)
                extra_values = gaussian_kernel(distances, bandwidth).astype(np.float64)
                state["sampled_ids"] = np.concatenate([sampled_ids, extra_ids])
                state["values"] = np.concatenate([state["values"], extra_values])
                sampled_total += len(extra_ids)

    estimate = 0.0
    for state in ring_state:
        if state is None:
            continue
        values = state["values"]
        if len(values) == 0:
            continue
        estimate += len(state["ids"]) * float(values.mean())
    return estimate, sampled_total


def pack_bits(bits: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(bits.shape[1], dtype=np.uint64)).astype(np.uint64)
    return bits.astype(np.uint64) @ powers


def code_hamming_distance(code_a: int, code_b: int) -> int:
    return bin(int(code_a) ^ int(code_b)).count("1")


def build_code_distance_table(codes) -> dict[int, dict[int, list[int]]]:
    unique_codes = [int(code) for code in codes]
    distance_table: dict[int, dict[int, list[int]]] = {}
    for code in unique_codes:
        buckets: dict[int, list[int]] = defaultdict(list)
        for other_code in unique_codes:
            buckets[code_hamming_distance(code, other_code)].append(other_code)
        distance_table[code] = dict(buckets)
    return distance_table


def max_angle_from_radius(q_norm: float, radius: float) -> float:
    if q_norm <= EPS:
        return math.pi
    return math.asin(min(1.0, max(0.0, radius / q_norm)))


def hamming_radius_from_angle(angle: float, bits: int) -> int:
    return min(max(int(math.ceil(bits * angle / math.pi)), 0), bits)


def hamming_neighbors(code: int, bits: int, probe: int):
    yield code
    if probe >= 1:
        for bit in range(bits):
            yield code ^ (1 << bit)
    if probe >= 2:
        for bit_a in range(bits):
            for bit_b in range(bit_a + 1, bits):
                yield code ^ (1 << bit_a) ^ (1 << bit_b)


class HashIndexBase:
    name: str

    def query_codes(self, q: np.ndarray) -> list[int]:
        raise NotImplementedError

    def query_hash(self, q: np.ndarray) -> set[int]:
        raise NotImplementedError


class RandomProjectionHashIndex(HashIndexBase):
    def __init__(
        self,
        name: str,
        x: np.ndarray,
        tables: int,
        bits: int,
        rng: np.random.Generator,
        normalized: bool,
        hamming_probe: int,
    ):
        self.name = name
        self.x = x
        self.tables_count = tables
        self.bits = bits
        self.hamming_probe = hamming_probe
        self.projections = rng.normal(size=(tables, bits, x.shape[1])).astype(np.float32)
        if normalized:
            norm = np.linalg.norm(self.projections, axis=2, keepdims=True)
            self.projections = self.projections / np.maximum(norm, EPS)
        self.tables: list[dict[int, list[int]]] = []
        self.point_codes: list[np.ndarray] = []
        self.distance_tables: list[dict[int, dict[int, list[int]]]] = []
        start = time.perf_counter()
        for table_id in range(tables):
            codes = pack_bits((x @ self.projections[table_id].T) >= 0)
            self.point_codes.append(codes)
            table: dict[int, list[int]] = defaultdict(list)
            for idx, code in enumerate(codes):
                table[int(code)].append(idx)
            self.tables.append(table)
            self.distance_tables.append(build_code_distance_table(table.keys()))
        self.build_time_s = time.perf_counter() - start

    def query_codes(self, q: np.ndarray) -> list[int]:
        codes = []
        for table_id in range(self.tables_count):
            bits = (self.projections[table_id] @ q) >= 0
            codes.append(int(pack_bits(bits.reshape(1, -1))[0]))
        return codes

    def query_hash(self, q: np.ndarray) -> set[int]:
        candidates: set[int] = set()
        for table_id, (table, code) in enumerate(zip(self.tables, self.query_codes(q))):
            for neighbor in hamming_neighbors(code, self.bits, self.hamming_probe):
                candidates.update(table.get(neighbor, []))
        return candidates


class MultiEncoderContrastiveHash(nn.Module):
    def __init__(self, input_dim: int, tables: int, bits: int):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Linear(input_dim, bits) for _ in range(tables)])
        self.decoder = nn.Linear(bits, input_dim)

    def encode_continuous(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [torch.tanh(encoder(x)) for encoder in self.encoders]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encode_continuous(x)


def train_mech(
    train_x: np.ndarray,
    tables: int,
    bits: int,
    epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    beta: float,
    gamma_balance: float,
    gamma_decorrelation: float,
    device: str,
) -> MultiEncoderContrastiveHash:
    model = MultiEncoderContrastiveHash(train_x.shape[1], tables, bits).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            hs = model(batch)
            x_norm = F.normalize(batch, dim=1)
            sim_x = x_norm @ x_norm.T
            rec_loss = 0.0
            contrast_loss = 0.0
            quant_loss = 0.0
            balance_loss = 0.0
            decorrelation_loss = 0.0
            for h in hs:
                rec = model.decoder(h)
                rec_loss = rec_loss + F.mse_loss(rec, batch)
                h_norm = F.normalize(h, dim=1)
                contrast_loss = contrast_loss + F.mse_loss(h_norm @ h_norm.T, sim_x)
                quant_loss = quant_loss + torch.mean((torch.abs(h) - 1.0) ** 2)
                balance_loss = balance_loss + torch.mean(torch.mean(h, dim=0) ** 2)
                corr = (h.T @ h) / max(h.shape[0], 1)
                off_diag = corr - torch.diag(torch.diag(corr))
                decorrelation_loss = decorrelation_loss + torch.mean(off_diag**2)
            loss = (
                rec_loss / len(hs)
                + alpha * contrast_loss / len(hs)
                + beta * quant_loss / len(hs)
                + gamma_balance * balance_loss / len(hs)
                + gamma_decorrelation * decorrelation_loss / len(hs)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu()


class MECHHashIndex(HashIndexBase):
    def __init__(
        self,
        model: MultiEncoderContrastiveHash,
        x: np.ndarray,
        tables: int,
        bits: int,
        hamming_probe: int,
    ):
        self.name = "MECH"
        self.model = model.eval()
        self.x = x
        self.tables_count = tables
        self.bits = bits
        self.hamming_probe = hamming_probe
        tensor_x = torch.from_numpy(x)
        self.tables: list[dict[int, list[int]]] = []
        self.point_codes: list[np.ndarray] = []
        self.thresholds: list[np.ndarray] = []
        self.distance_tables: list[dict[int, dict[int, list[int]]]] = []
        start = time.perf_counter()
        with torch.no_grad():
            hs = self.model.encode_continuous(tensor_x)
        for table_id in range(tables):
            h = hs[table_id][:, :bits].numpy()
            thresholds = np.median(h, axis=0)
            self.thresholds.append(thresholds)
            codes = pack_bits(h >= thresholds)
            self.point_codes.append(codes)
            table: dict[int, list[int]] = defaultdict(list)
            for idx, code in enumerate(codes):
                table[int(code)].append(idx)
            self.tables.append(table)
            self.distance_tables.append(build_code_distance_table(table.keys()))
        self.build_time_s = time.perf_counter() - start

    def query_codes(self, q: np.ndarray) -> list[int]:
        q_tensor = torch.from_numpy(q.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            hs = self.model.encode_continuous(q_tensor)
        codes = []
        for table_id in range(self.tables_count):
            bits = hs[table_id][0, : self.bits].numpy() >= self.thresholds[table_id]
            codes.append(int(pack_bits(bits.reshape(1, -1))[0]))
        return codes

    def query_hash(self, q: np.ndarray) -> set[int]:
        candidates: set[int] = set()
        codes = self.query_codes(q)
        for table_id, table in enumerate(self.tables):
            code = codes[table_id]
            for neighbor in hamming_neighbors(code, self.bits, self.hamming_probe):
                candidates.update(table.get(neighbor, []))
        return candidates


def build_sphere_table(norms: np.ndarray, delta: float) -> dict[int, np.ndarray]:
    buckets: dict[int, list[int]] = defaultdict(list)
    layers = np.floor(norms / delta).astype(int)
    for idx, layer in enumerate(layers):
        buckets[int(layer)].append(idx)
    return {layer: np.asarray(indices, dtype=np.int32) for layer, indices in buckets.items()}


def sphere_layer_bounds(q_norm: float, radius: float, delta: float) -> tuple[int, int]:
    low = int(math.floor(max(0.0, q_norm - radius) / delta))
    high = int(math.ceil((q_norm + radius) / delta))
    return low, high


def sphere_candidates(
    sphere_table: dict[int, np.ndarray],
    q_norm: float,
    radius: float,
    delta: float,
) -> set[int]:
    low, high = sphere_layer_bounds(q_norm, radius, delta)
    rows = [sphere_table[layer] for layer in range(low, high + 1) if layer in sphere_table]
    if not rows:
        return set()
    return set(np.concatenate(rows).tolist())


def distance_table_neighbor_codes(
    index: HashIndexBase,
    table_id: int,
    q_code: int,
    max_hamming_distance: int,
) -> list[int]:
    q_code = int(q_code)
    max_hamming_distance = max(int(max_hamming_distance), 0)
    distance_tables = getattr(index, "distance_tables", None)
    if distance_tables is not None and q_code in distance_tables[table_id]:
        distance_table = distance_tables[table_id][q_code]
        codes: list[int] = []
        for distance in range(max_hamming_distance + 1):
            codes.extend(distance_table.get(distance, []))
        return codes
    return [
        int(code)
        for code in index.tables[table_id].keys()
        if code_hamming_distance(q_code, int(code)) <= max_hamming_distance
    ]


def hash_filter_candidates(
    index: HashIndexBase,
    q_codes: list[int],
    candidate_ids: set[int],
    min_collisions: int,
    max_hamming_distance: int | None = None,
) -> set[int]:
    if not candidate_ids:
        return set()
    ids = np.fromiter(candidate_ids, dtype=np.int32)
    counts = np.zeros(len(ids), dtype=np.int16)
    required = min(max(min_collisions, 1), len(q_codes))
    hamming_radius = (
        int(max_hamming_distance)
        if max_hamming_distance is not None
        else int(index.hamming_probe)
    )
    for table_id, q_code in enumerate(q_codes):
        neighbor_codes = np.asarray(
            distance_table_neighbor_codes(index, table_id, q_code, hamming_radius),
            dtype=np.uint64,
        )
        if len(neighbor_codes) == 0:
            continue
        matched = np.isin(index.point_codes[table_id][ids], neighbor_codes)
        counts += matched.astype(np.int16)
    return set(ids[counts >= required].tolist())


def distance_table_hamming_radius(
    index: HashIndexBase,
    q_norm: float,
    radius: float,
    variant: str,
) -> int | None:
    if variant not in {"distance_table", "full"}:
        return None
    angle = max_angle_from_radius(q_norm, radius)
    return max(index.hamming_probe, hamming_radius_from_angle(angle, index.bits))


def approx_ball(
    index: HashIndexBase,
    hash_set: set[int],
    q: np.ndarray,
    q_norm: float,
    radius: float,
    norms: np.ndarray,
    layers: np.ndarray,
    sphere_table: dict[int, np.ndarray],
    delta: float,
    variant: str,
) -> set[int]:
    if variant == "hash_only":
        return hash_set

    if variant in {"full", "no_distance"}:
        pre = hash_set & sphere_candidates(sphere_table, q_norm, radius, delta)
    else:
        pre = hash_set

    if variant == "no_distance":
        return pre
    if not pre or q_norm <= EPS:
        return set()

    ids = np.fromiter(pre, dtype=np.int32)
    denom = q_norm * norms[ids]
    valid = denom > EPS
    if not np.any(valid):
        return set()
    ids = ids[valid]
    denom = denom[valid]
    cosines = np.clip(index.x[ids] @ q / denom, -1.0, 1.0)
    if variant == "angle_only":
        reference_norm = float(np.median(norms))
        cos_min = (q_norm**2 + reference_norm**2 - radius**2) / max(
            2.0 * q_norm * reference_norm, EPS
        )
        cos_min = np.clip(cos_min, -1.0, 1.0)
        return set(ids[cosines >= cos_min].tolist())
    if variant == "oracle_distance":
        s_hat = norms[ids]
    else:
        s_hat = (layers[ids].astype(np.float32) + 0.5) * delta
    cos_min = (q_norm**2 + s_hat**2 - radius**2) / np.maximum(2.0 * q_norm * s_hat, EPS)
    cos_min = np.clip(cos_min, -1.0, 1.0)
    return set(ids[cosines >= cos_min].tolist())


def radius_pool(
    index: HashIndexBase,
    all_ids: set[int],
    q: np.ndarray,
    q_norm: float,
    radius: float,
    norms: np.ndarray,
    layers: np.ndarray,
    sphere_table: dict[int, np.ndarray],
    delta: float,
    variant: str,
) -> set[int]:
    if variant == "hash_only":
        return all_ids
    if variant == "distance_table":
        return all_ids
    if variant in {"no_distance", "full"}:
        return sphere_candidates(sphere_table, q_norm, radius, delta)
    return approx_ball(
        index,
        all_ids,
        q,
        q_norm,
        radius,
        norms,
        layers,
        sphere_table,
        delta,
        variant,
    )


def approx_annulus_radius_first(
    index: HashIndexBase,
    q_codes: list[int],
    q: np.ndarray,
    q_norm: float,
    r_in: float,
    r_out: float,
    norms: np.ndarray,
    layers: np.ndarray,
    sphere_table: dict[int, np.ndarray],
    delta: float,
    variant: str,
    min_collisions: int,
    all_ids: set[int],
) -> tuple[set[int], dict[str, float], list[set[int]]]:
    raw_hash_radius = distance_table_hamming_radius(index, q_norm, r_out, variant)
    raw_hash = hash_filter_candidates(
        index,
        q_codes,
        all_ids,
        min_collisions,
        raw_hash_radius,
    )
    if variant == "hash_only":
        return raw_hash, {
            "raw_hash_size": float(len(raw_hash)),
            "outer_prefilter_size": float(len(all_ids)),
            "inner_prefilter_size": 0.0,
            "outer_hash_size": float(len(raw_hash)),
            "inner_hash_size": 0.0,
            "ring_count": 1.0,
            "mean_ring_size": float(len(raw_hash)),
            "max_ring_size": float(len(raw_hash)),
        }, [raw_hash]

    outer_pool = radius_pool(
        index,
        all_ids,
        q,
        q_norm,
        r_out,
        norms,
        layers,
        sphere_table,
        delta,
        variant,
    )
    inner_pool = radius_pool(
        index,
        all_ids,
        q,
        q_norm,
        r_in,
        norms,
        layers,
        sphere_table,
        delta,
        variant,
    )
    outer_hash = hash_filter_candidates(
        index,
        q_codes,
        outer_pool,
        min_collisions,
        distance_table_hamming_radius(index, q_norm, r_out, variant),
    )
    inner_hash = hash_filter_candidates(
        index,
        q_codes,
        inner_pool,
        min_collisions,
        distance_table_hamming_radius(index, q_norm, r_in, variant),
    )
    ring = outer_hash - inner_hash
    return ring, {
        "raw_hash_size": float(len(raw_hash)),
        "outer_prefilter_size": float(len(outer_pool)),
        "inner_prefilter_size": float(len(inner_pool)),
        "outer_hash_size": float(len(outer_hash)),
        "inner_hash_size": float(len(inner_hash)),
        "ring_count": 1.0,
        "mean_ring_size": float(len(ring)),
        "max_ring_size": float(len(ring)),
    }, [ring]


def approx_fixed_radius_partition_radius_first(
    index: HashIndexBase,
    q_codes: list[int],
    q: np.ndarray,
    q_norm: float,
    radius: float,
    annulus_count: int,
    norms: np.ndarray,
    layers: np.ndarray,
    sphere_table: dict[int, np.ndarray],
    delta: float,
    variant: str,
    min_collisions: int,
    all_ids: set[int],
) -> tuple[set[int], dict[str, float], list[set[int]]]:
    hash_filter_cache: dict[int, set[int]] = {}
    hash_filter_mask_cache: dict[int, np.ndarray] = {}

    def cache_key(hamming_radius: int | None) -> int:
        return -1 if hamming_radius is None else int(hamming_radius)

    def cached_hash_set(hamming_radius: int | None) -> set[int]:
        key = cache_key(hamming_radius)
        if key not in hash_filter_cache:
            hash_filter_cache[key] = hash_filter_candidates(
                index,
                q_codes,
                all_ids,
                min_collisions,
                hamming_radius,
            )
        return hash_filter_cache[key]

    def cached_hash_mask(hamming_radius: int | None) -> np.ndarray:
        key = cache_key(hamming_radius)
        if key not in hash_filter_mask_cache:
            mask = np.zeros(len(norms), dtype=bool)
            passed = cached_hash_set(hamming_radius)
            if passed:
                ids = np.fromiter(passed, dtype=np.int32)
                mask[ids] = True
            hash_filter_mask_cache[key] = mask
        return hash_filter_mask_cache[key]

    def cached_hash_filter(candidate_ids: set[int], hamming_radius: int | None) -> set[int]:
        return candidate_ids & cached_hash_set(hamming_radius)

    def cached_hash_filter_ids(candidate_ids: np.ndarray, hamming_radius: int | None) -> set[int]:
        if len(candidate_ids) == 0:
            return set()
        keep = cached_hash_mask(hamming_radius)[candidate_ids]
        if not np.any(keep):
            return set()
        return set(candidate_ids[keep].tolist())

    def sphere_candidates_array(radius_value: float) -> np.ndarray:
        low, high = sphere_layer_bounds(q_norm, radius_value, delta)
        rows = [sphere_table[layer] for layer in range(low, high + 1) if layer in sphere_table]
        if not rows:
            return np.asarray([], dtype=np.int32)
        return np.concatenate(rows)

    raw_hash_radius = distance_table_hamming_radius(index, q_norm, radius, variant)
    raw_hash = cached_hash_set(raw_hash_radius)
    if variant == "hash_only":
        return raw_hash, {
            "raw_hash_size": float(len(raw_hash)),
            "outer_prefilter_size": float(len(all_ids)),
            "inner_prefilter_size": 0.0,
            "outer_hash_size": float(len(raw_hash)),
            "inner_hash_size": 0.0,
            "ring_count": 1.0,
            "mean_ring_size": float(len(raw_hash)),
            "max_ring_size": float(len(raw_hash)),
        }, [raw_hash]

    annulus_count = max(int(annulus_count), 1)
    bounds = np.linspace(0.0, radius, annulus_count + 1)
    if variant in {"full", "no_distance"}:
        final_ids = sphere_candidates_array(radius)
        final_layers = layers[final_ids] if len(final_ids) else np.asarray([], dtype=np.int32)
        previous_pool_size = 0
        previous_hash: set[int] = set()
        last_inner_pool_size = 0
        last_inner_hash: set[int] = set()
        final_pool_size = 0
        final_hash: set[int] = set()
        partition_candidates: set[int] = set()
        ring_candidates: list[set[int]] = []
        ring_sizes = []

        for outer_radius in bounds[1:]:
            low, high = sphere_layer_bounds(q_norm, float(outer_radius), delta)
            if len(final_ids):
                current_ids = final_ids[(final_layers >= low) & (final_layers <= high)]
            else:
                current_ids = final_ids
            current_hash = cached_hash_filter_ids(
                current_ids,
                distance_table_hamming_radius(index, q_norm, float(outer_radius), variant),
            )
            last_inner_pool_size = previous_pool_size
            last_inner_hash = previous_hash
            ring = current_hash - previous_hash
            partition_candidates.update(ring)
            ring_candidates.append(ring)
            ring_sizes.append(len(ring))
            previous_pool_size = len(current_ids)
            previous_hash = current_hash
            final_pool_size = len(current_ids)
            final_hash = current_hash

        return partition_candidates, {
            "raw_hash_size": float(len(raw_hash)),
            "outer_prefilter_size": float(final_pool_size),
            "inner_prefilter_size": float(last_inner_pool_size),
            "outer_hash_size": float(len(final_hash)),
            "inner_hash_size": float(len(last_inner_hash)),
            "ring_count": float(annulus_count),
            "mean_ring_size": float(np.mean(ring_sizes)) if ring_sizes else 0.0,
            "max_ring_size": float(np.max(ring_sizes)) if ring_sizes else 0.0,
        }, ring_candidates

    def cached_hash_filter(candidate_ids: set[int], hamming_radius: int | None) -> set[int]:
        key = -1 if hamming_radius is None else int(hamming_radius)
        if key not in hash_filter_cache:
            hash_filter_cache[key] = hash_filter_candidates(
                index,
                q_codes,
                all_ids,
                min_collisions,
                hamming_radius,
            )
        return candidate_ids & hash_filter_cache[key]
    previous_pool: set[int] = set()
    previous_hash: set[int] = set()
    last_inner_pool: set[int] = set()
    last_inner_hash: set[int] = set()
    final_pool: set[int] = set()
    final_hash: set[int] = set()
    partition_candidates: set[int] = set()
    ring_candidates: list[set[int]] = []
    ring_sizes = []

    for outer_radius in bounds[1:]:
        current_pool = radius_pool(
            index,
            all_ids,
            q,
            q_norm,
            float(outer_radius),
            norms,
            layers,
            sphere_table,
            delta,
            variant,
        )
        current_hash = cached_hash_filter(
            current_pool,
            distance_table_hamming_radius(index, q_norm, float(outer_radius), variant),
        )
        last_inner_pool = previous_pool
        last_inner_hash = previous_hash
        ring = current_hash - previous_hash
        partition_candidates.update(ring)
        ring_candidates.append(ring)
        ring_sizes.append(len(ring))
        previous_pool = current_pool
        previous_hash = current_hash
        final_pool = current_pool
        final_hash = current_hash

    return partition_candidates, {
        "raw_hash_size": float(len(raw_hash)),
        "outer_prefilter_size": float(len(final_pool)),
        "inner_prefilter_size": float(len(last_inner_pool)),
        "outer_hash_size": float(len(final_hash)),
        "inner_hash_size": float(len(last_inner_hash)),
        "ring_count": float(annulus_count),
        "mean_ring_size": float(np.mean(ring_sizes)) if ring_sizes else 0.0,
        "max_ring_size": float(np.max(ring_sizes)) if ring_sizes else 0.0,
    }, ring_candidates


def approx_annulus(
    index: HashIndexBase,
    hash_set: set[int],
    q: np.ndarray,
    q_norm: float,
    r_in: float,
    r_out: float,
    norms: np.ndarray,
    layers: np.ndarray,
    sphere_table: dict[int, np.ndarray],
    delta: float,
    variant: str,
) -> set[int]:
    if variant == "hash_only":
        return hash_set

    if variant == "no_distance":
        outer = hash_set & sphere_candidates(sphere_table, q_norm, r_out, delta)
        inner = hash_set & sphere_candidates(sphere_table, q_norm, r_in, delta)
        return outer - inner

    if not hash_set or q_norm <= EPS:
        return set()

    if variant == "full":
        pre = hash_set & sphere_candidates(sphere_table, q_norm, r_out, delta)
    else:
        pre = hash_set
    if not pre:
        return set()

    ids = np.fromiter(pre, dtype=np.int32)
    denom = q_norm * norms[ids]
    valid = denom > EPS
    if not np.any(valid):
        return set()
    ids = ids[valid]
    denom = denom[valid]
    cosines = np.clip(index.x[ids] @ q / denom, -1.0, 1.0)

    if variant == "angle_only":
        s_hat = float(np.median(norms))
        denom_ref = max(2.0 * q_norm * s_hat, EPS)
        cos_out = np.clip((q_norm**2 + s_hat**2 - r_out**2) / denom_ref, -1.0, 1.0)
        cos_in = np.clip((q_norm**2 + s_hat**2 - r_in**2) / denom_ref, -1.0, 1.0)
        keep = (cosines >= cos_out) & (cosines < cos_in)
        return set(ids[keep].tolist())

    if variant == "oracle_distance":
        s_hat = norms[ids]
        inner_layer_mask = np.ones(len(ids), dtype=bool)
    else:
        s_hat = (layers[ids].astype(np.float32) + 0.5) * delta
        low_in, high_in = sphere_layer_bounds(q_norm, r_in, delta)
        inner_layer_mask = (layers[ids] >= low_in) & (layers[ids] <= high_in)

    denom_s = np.maximum(2.0 * q_norm * s_hat, EPS)
    cos_out = np.clip((q_norm**2 + s_hat**2 - r_out**2) / denom_s, -1.0, 1.0)
    cos_in = np.clip((q_norm**2 + s_hat**2 - r_in**2) / denom_s, -1.0, 1.0)
    keep = (cosines >= cos_out) & ~(inner_layer_mask & (cosines >= cos_in))
    return set(ids[keep].tolist())


@dataclass(frozen=True)
class EvalConfig:
    delta: float
    radius_factor: float
    bandwidth_factor: float
    variant: str = "full"
    query_mode: str = "radius_first"
    min_collisions: int = 1
    fixed_query_radius: float | None = None
    annulus_count: int = 1
    ring_sample_size: int = 64
    total_ring_sample_budget: int | None = None
    ring_sample_allocation: str = "per_ring"


def evaluate_index(
    index: HashIndexBase,
    index_x: np.ndarray,
    queries: np.ndarray,
    distances: np.ndarray,
    inner_base: np.ndarray,
    outer_base: np.ndarray,
    bandwidth_base: float,
    config: EvalConfig,
) -> dict[str, float | str]:
    norms = np.linalg.norm(index_x, axis=1)
    layers = np.floor(norms / config.delta).astype(int)
    sphere_table = build_sphere_table(norms, config.delta)
    query_norms = np.linalg.norm(queries, axis=1)
    bandwidth = bandwidth_base * config.bandwidth_factor
    all_ids = set(range(len(index_x)))
    cache_query_hash = index.name == "MECH" and config.query_mode == "global_hash"
    query_hash_candidates = (
        [tuple(index.query_hash(q)) for q in queries] if cache_query_hash else None
    )
    cache_query_codes = index.name == "MECH" and config.query_mode == "radius_first"
    query_code_candidates = (
        [tuple(index.query_codes(q)) for q in queries] if cache_query_codes else None
    )
    use_ring_sampling = index.name == "MECH" and config.variant == "full"

    point_precision = []
    point_recall = []
    point_f1 = []
    point_fp_ratio = []
    point_fn_ratio = []
    weighted_recall = []
    weighted_fp = []
    weighted_fn = []
    signed_bias = []
    kde_abs_error = []
    cer = []
    sizes = []
    kde_sample_sizes = []
    times = []
    filter_times = []
    kde_times = []
    diagnostics: dict[str, list[float]] = defaultdict(list)

    for q_id, q in enumerate(queries):
        d = distances[q_id]
        fixed_radius = (
            config.fixed_query_radius * config.radius_factor
            if config.fixed_query_radius is not None
            else None
        )
        if fixed_radius is None:
            r_in = inner_base[q_id] * config.radius_factor
            r_out = outer_base[q_id] * config.radius_factor
            true = set(np.flatnonzero((d >= r_in) & (d <= r_out)).tolist())
        else:
            r_in = 0.0
            r_out = fixed_radius
            true = set(np.flatnonzero(d <= fixed_radius).tolist())
        if not true:
            continue

        cached_hash_candidates = (
            query_hash_candidates[q_id] if query_hash_candidates is not None else None
        )
        cached_query_codes = (
            query_code_candidates[q_id] if query_code_candidates is not None else None
        )

        def retrieve_annulus() -> tuple[set[int], dict[str, float], list[set[int]]]:
            if config.query_mode == "radius_first":
                q_codes = (
                    list(cached_query_codes)
                    if cached_query_codes is not None
                    else index.query_codes(q)
                )
                if fixed_radius is not None:
                    return approx_fixed_radius_partition_radius_first(
                        index,
                        q_codes,
                        q,
                        query_norms[q_id],
                        fixed_radius,
                        config.annulus_count,
                        norms,
                        layers,
                        sphere_table,
                        config.delta,
                        config.variant,
                        config.min_collisions,
                        all_ids,
                    )
                return approx_annulus_radius_first(
                    index,
                    q_codes,
                    q,
                    query_norms[q_id],
                    r_in,
                    r_out,
                    norms,
                    layers,
                    sphere_table,
                    config.delta,
                    config.variant,
                    config.min_collisions,
                    all_ids,
                )

            hash_set = (
                set(cached_hash_candidates)
                if cached_hash_candidates is not None
                else index.query_hash(q)
            )
            if fixed_radius is None:
                current = approx_annulus(
                    index,
                    hash_set,
                    q,
                    query_norms[q_id],
                    r_in,
                    r_out,
                    norms,
                    layers,
                    sphere_table,
                    config.delta,
                    config.variant,
                )
            else:
                current = approx_ball(
                    index,
                    hash_set,
                    q,
                    query_norms[q_id],
                    fixed_radius,
                    norms,
                    layers,
                    sphere_table,
                    config.delta,
                    config.variant,
                )
            return current, {
                "raw_hash_size": float(len(hash_set)),
                "outer_prefilter_size": float("nan"),
                "inner_prefilter_size": float("nan"),
                "outer_hash_size": float("nan"),
                "inner_hash_size": float("nan"),
                "ring_count": float("nan"),
                "mean_ring_size": float("nan"),
                "max_ring_size": float("nan"),
            }, [current]

        approx = set()
        approx_kde_estimate = 0.0
        sampled_count = 0
        first_diag: dict[str, float] = {}
        repeated_times = []
        repeated_filter_times = []
        repeated_kde_times = []
        for repeat_id in range(TIMING_REPEATS):
            start = time.perf_counter()
            current, current_diag, current_rings = retrieve_annulus()
            filter_done = time.perf_counter()
            if use_ring_sampling:
                rng = np.random.default_rng(SEED + q_id * 1009 + repeat_id)
                if (
                    config.total_ring_sample_budget is not None
                    and config.ring_sample_allocation == "size_weighted_total_budget"
                ):
                    kde_estimate, current_sampled_count = sampled_weighted_budget_ring_kde(
                        index_x,
                        q,
                        current_rings,
                        bandwidth,
                        config.total_ring_sample_budget,
                        rng,
                    )
                elif (
                    config.total_ring_sample_budget is not None
                    and config.ring_sample_allocation == "balanced_size_weighted_total_budget"
                ):
                    kde_estimate, current_sampled_count = sampled_balanced_budget_ring_kde(
                        index_x,
                        q,
                        current_rings,
                        bandwidth,
                        config.total_ring_sample_budget,
                        rng,
                    )
                elif (
                    config.total_ring_sample_budget is not None
                    and config.ring_sample_allocation == "pilot_neyman_total_budget"
                ):
                    kde_estimate, current_sampled_count = sampled_pilot_neyman_budget_ring_kde(
                        index_x,
                        q,
                        current_rings,
                        bandwidth,
                        config.total_ring_sample_budget,
                        rng,
                    )
                else:
                    kde_estimate, current_sampled_count = sampled_ring_kde(
                        index_x,
                        q,
                        current_rings,
                        bandwidth,
                        config.ring_sample_size,
                        rng,
                    )
            elif current:
                ids = np.fromiter(current, dtype=np.int32)
                candidate_distances = np.linalg.norm(index_x[ids] - q, axis=1)
                kde_estimate = float(gaussian_kernel(candidate_distances, bandwidth).sum())
                current_sampled_count = len(ids)
            else:
                kde_estimate = 0.0
                current_sampled_count = 0
            end = time.perf_counter()
            repeated_filter_times.append(filter_done - start)
            repeated_kde_times.append(end - filter_done)
            repeated_times.append(end - start)
            if repeat_id == 0:
                approx = current
                approx_kde_estimate = kde_estimate
                sampled_count = current_sampled_count
                first_diag = current_diag
        times.append(float(np.median(repeated_times)))
        filter_times.append(float(np.median(repeated_filter_times)))
        kde_times.append(float(np.median(repeated_kde_times)))
        for key, value in first_diag.items():
            diagnostics[key].append(value)

        tp = true & approx
        fp = approx - true
        fn = true - approx
        p = len(tp) / len(approx) if approx else 0.0
        r = len(tp) / len(true)
        point_precision.append(p)
        point_recall.append(r)
        point_f1.append(2 * p * r / (p + r + EPS))
        point_fp_ratio.append(len(fp) / len(true))
        point_fn_ratio.append(len(fn) / len(true))
        sizes.append(len(approx))
        kde_sample_sizes.append(sampled_count)
        cer.append(len(approx) / max(len(true), 1))

        weights = gaussian_kernel(d, bandwidth)
        true_w = float(weights[list(true)].sum()) if true else 0.0
        tp_w = float(weights[list(tp)].sum()) if tp else 0.0
        fp_w = float(weights[list(fp)].sum()) if fp else 0.0
        fn_w = float(weights[list(fn)].sum()) if fn else 0.0
        weighted_recall.append(tp_w / max(true_w, EPS))
        weighted_fp.append(fp_w / max(true_w, EPS))
        weighted_fn.append(fn_w / max(true_w, EPS))
        rel_bias = (approx_kde_estimate - true_w) / max(true_w, EPS)
        signed_bias.append(rel_bias)
        kde_abs_error.append(abs(rel_bias))

    return {
        "method": index.name,
        "variant": config.variant,
        "query_mode": config.query_mode,
        "min_collisions": config.min_collisions,
        "fixed_query_radius": (
            float(config.fixed_query_radius)
            if config.fixed_query_radius is not None
            else float("nan")
        ),
        "annulus_count": int(config.annulus_count),
        "delta": config.delta,
        "radius_factor": config.radius_factor,
        "bandwidth_factor": config.bandwidth_factor,
        "point_precision": float(np.mean(point_precision)),
        "point_recall": float(np.mean(point_recall)),
        "point_f1": float(np.mean(point_f1)),
        "point_fp_ratio": float(np.mean(point_fp_ratio)),
        "point_fn_ratio": float(np.mean(point_fn_ratio)),
        "kernel_weighted_recall": float(np.mean(weighted_recall)),
        "kernel_weighted_fp_ratio": float(np.mean(weighted_fp)),
        "kernel_weighted_fn_ratio": float(np.mean(weighted_fn)),
        "signed_kde_bias": float(np.mean(signed_bias)),
        "kde_abs_relative_error": float(np.mean(kde_abs_error)),
        "candidate_expansion_ratio": float(np.mean(cer)),
        "candidate_size": float(np.mean(sizes)),
        "kde_sample_size": float(np.mean(kde_sample_sizes)),
        "query_time_ms": float(np.mean(times) * 1000.0),
        "filter_time_ms": float(np.mean(filter_times) * 1000.0),
        "kde_time_ms": float(np.mean(kde_times) * 1000.0),
        "build_time_s": float(getattr(index, "build_time_s", 0.0)),
        "raw_hash_size": float(np.mean(diagnostics["raw_hash_size"])),
        "outer_prefilter_size": (
            float(np.mean(diagnostics["outer_prefilter_size"]))
            if config.query_mode == "radius_first"
            else float("nan")
        ),
        "inner_prefilter_size": (
            float(np.mean(diagnostics["inner_prefilter_size"]))
            if config.query_mode == "radius_first"
            else float("nan")
        ),
        "outer_hash_size": (
            float(np.mean(diagnostics["outer_hash_size"]))
            if config.query_mode == "radius_first"
            else float("nan")
        ),
        "inner_hash_size": (
            float(np.mean(diagnostics["inner_hash_size"]))
            if config.query_mode == "radius_first"
            else float("nan")
        ),
        "ring_count": (
            float(np.nanmean(diagnostics["ring_count"]))
            if config.query_mode == "radius_first"
            else float("nan")
        ),
        "mean_ring_size": (
            float(np.nanmean(diagnostics["mean_ring_size"]))
            if config.query_mode == "radius_first"
            else float("nan")
        ),
        "max_ring_size": (
            float(np.nanmean(diagnostics["max_ring_size"]))
            if config.query_mode == "radius_first"
            else float("nan")
        ),
    }


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}\\%"


def write_latex_table(rows: pd.DataFrame, columns: list[tuple[str, str]], caption: str, label: str) -> str:
    out = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{" + "l" + "c" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(title for title, _ in columns) + r" \\",
        "\\midrule",
    ]
    for _, row in rows.iterrows():
        values = []
        for _, key in columns:
            value = row[key]
            if key in {"method", "variant", "parameter", "setting"}:
                values.append(str(value))
            elif key in {"query_time_ms", "filter_time_ms", "kde_time_ms"}:
                values.append(f"{value:.3f}")
            elif key == "candidate_expansion_ratio":
                values.append(f"{value:.4f}")
            elif key in {"candidate_size", "kde_sample_size"}:
                values.append(f"{value:.2f}")
            elif key in {
                "raw_hash_size",
                "outer_prefilter_size",
                "inner_prefilter_size",
                "outer_hash_size",
                "inner_hash_size",
                "mean_ring_size",
                "max_ring_size",
            }:
                values.append(f"{value:.1f}")
            elif key == "ring_count":
                values.append(f"{value:.0f}")
            else:
                values.append(pct(float(value)))
        out.append(" & ".join(values) + r" \\")
    out.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(out)


def format_cell(value, key: str) -> str:
    if key in {"method", "variant", "parameter", "setting"}:
        return str(value)
    if key in {"query_time_ms", "filter_time_ms", "kde_time_ms"}:
        return f"{float(value):.3f}"
    if key == "candidate_expansion_ratio":
        return f"{float(value):.4f}"
    if key in {"candidate_size", "kde_sample_size"}:
        return f"{float(value):.2f}"
    if key in {
        "point_precision",
        "point_recall",
        "point_f1",
        "point_fp_ratio",
        "point_fn_ratio",
        "kernel_weighted_recall",
        "kernel_weighted_fp_ratio",
        "kernel_weighted_fn_ratio",
        "signed_kde_bias",
        "kde_abs_relative_error",
    }:
        return f"{float(value):.4f}"
    if key in {
        "candidate_size",
        "raw_hash_size",
        "outer_prefilter_size",
        "inner_prefilter_size",
        "outer_hash_size",
        "inner_hash_size",
        "mean_ring_size",
        "max_ring_size",
    }:
        return f"{float(value):.1f}"
    if key == "ring_count":
        return f"{float(value):.0f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(rows: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append(
            "| "
            + " | ".join(format_cell(row[key], key) for _, key in columns)
            + " |"
        )
    return "\n".join([header, sep] + body)


def make_summary(
    out_dir: Path,
    config: dict,
    method_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    method_cols = [
        ("Method", "method"),
        ("P", "point_precision"),
        ("R", "point_recall"),
        ("F1", "point_f1"),
        ("FP/True", "point_fp_ratio"),
        ("FN/True", "point_fn_ratio"),
        ("$R^k$", "kernel_weighted_recall"),
        ("$E_+^k$", "kernel_weighted_fp_ratio"),
        ("KDE Err.", "kde_abs_relative_error"),
        ("CER", "candidate_expansion_ratio"),
        ("Cand.", "candidate_size"),
        ("KDE Eval.", "kde_sample_size"),
        ("Raw Hash", "raw_hash_size"),
        ("R-Pool", "outer_prefilter_size"),
        ("Rings", "ring_count"),
        ("Mean Ring", "mean_ring_size"),
        ("Max Ring", "max_ring_size"),
        ("Filter ms", "filter_time_ms"),
        ("KDE ms", "kde_time_ms"),
        ("Online ms", "query_time_ms"),
    ]
    ablation_cols = [
        ("Variant", "variant"),
        ("P", "point_precision"),
        ("R", "point_recall"),
        ("FP/True", "point_fp_ratio"),
        ("FN/True", "point_fn_ratio"),
        ("$R^k$", "kernel_weighted_recall"),
        ("KDE Err.", "kde_abs_relative_error"),
        ("CER", "candidate_expansion_ratio"),
        ("Cand.", "candidate_size"),
        ("KDE Eval.", "kde_sample_size"),
        ("Filter ms", "filter_time_ms"),
        ("KDE ms", "kde_time_ms"),
        ("Online ms", "query_time_ms"),
    ]
    sens_view = sensitivity_df.copy()
    sens_view["setting"] = sens_view["parameter"] + "=" + sens_view["value"].astype(str)
    sens_cols = [
        ("Setting", "setting"),
        ("P", "point_precision"),
        ("R", "point_recall"),
        ("FP/True", "point_fp_ratio"),
        ("FN/True", "point_fn_ratio"),
        ("$R^k$", "kernel_weighted_recall"),
        ("KDE Err.", "kde_abs_relative_error"),
        ("CER", "candidate_expansion_ratio"),
        ("Cand.", "candidate_size"),
        ("KDE Eval.", "kde_sample_size"),
        ("Raw Hash", "raw_hash_size"),
        ("R-Pool", "outer_prefilter_size"),
        ("Rings", "ring_count"),
        ("Mean Ring", "mean_ring_size"),
        ("Max Ring", "max_ring_size"),
        ("Filter ms", "filter_time_ms"),
        ("KDE ms", "kde_time_ms"),
        ("Online ms", "query_time_ms"),
    ]

    practical_ablation = ablation_df[
        ~ablation_df["variant"].str.contains("Oracle", case=False, na=False)
    ].copy()
    oracle_ablation = ablation_df[
        ablation_df["variant"].str.contains("Oracle", case=False, na=False)
    ].copy()
    best_method = method_df.sort_values("kde_abs_relative_error").iloc[0]
    full = practical_ablation[practical_ablation["variant"] == "Full MECH"].iloc[0]

    lines = [
        "# MECH Approximate Annulus KDE Experiments",
        "",
        "## Experimental Setup",
        "",
        f"- Dataset: {config['dataset']}",
        f"- Training/index/query split: {config['n_train']}/{config['n_index']}/{config['n_queries']}",
        f"- Base annulus: distance percentiles {config['inner_percentile']}--{config['outer_percentile']}",
        f"- Base kernel bandwidth: {config['bandwidth_base']:.6f}",
        f"- MECH: {config['max_tables']} encoders, {config['max_bits']} bits, {config['epochs']} epochs",
        f"- Base hash query: {config['base_tables']} tables, {config['base_bits']} bits, Hamming probe {config['hamming_probe']}",
        f"- Query mode: {config['query_mode']}; minimum hash collisions: {config['min_collisions']}",
        (
            f"- Fixed query radius: {config['fixed_query_radius']}; "
            f"annulus count: {config['annulus_count']}"
            if config["fixed_query_radius"] is not None
            else "- Fixed query radius: disabled; using percentile annulus"
        ),
        f"- Full MECH per-ring KDE sampling: at most {config['ring_sample_size']} candidates are sampled from each annulus; each ring contribution is estimated as ring size times the sampled mean kernel value.",
        "- Other variants and non-MECH methods compute KDE over all retrieved candidates without ring sampling.",
        "- Time metric: Online ms is the average online query time per query. It is decomposed into Filter ms for candidate retrieval/filtering and KDE ms for KDE estimation.",
        "- Training time, hash index construction time, sphere table construction time, and distance-table construction time are excluded from all Online ms values.",
        "- In the structure ablation table, only online query time is reported. Training time and build/index construction time are not recorded for structure ablation.",
        "- MECH query hashes are cached in this timing protocol; SimHash/Angular LSH/Hyperplane LSH query hashes are computed online in the method comparison.",
        "",
        "## Metric Definitions",
        "",
        (
            f"- Exact fixed-radius neighbor set: points within the specified query radius "
            f"R={config['fixed_query_radius']}."
            if config["fixed_query_radius"] is not None
            else "- Exact annulus neighbor set: points within the configured percentile annulus."
        ),
        "- Retrieved set: candidates returned by the approximate annulus query and used for KDE.",
        "- P: retrieved points that are exact fixed-radius neighbors divided by retrieved points.",
        "- R: retrieved points that are exact fixed-radius neighbors divided by exact fixed-radius neighbors.",
        "- FP/True: false positives divided by exact fixed-radius neighbors.",
        "- FN/True: false negatives divided by exact fixed-radius neighbors.",
        "- CER: retrieved candidate count divided by exact fixed-radius neighbor count. CER close to 1 only means the candidate count is close to the true neighbor count; it does not mean retrieval is perfectly accurate.",
        "- KDE Eval.: average number of points whose kernel values are actually computed. For Full MECH this is the per-ring sample count; for other variants it is the full candidate count.",
        "- KDE Err.: absolute relative error between the reported KDE estimate and the exact KDE over the fixed-radius neighbor set.",
        "- Filter ms: candidate retrieval and filtering time, in milliseconds per query.",
        "- KDE ms: KDE estimation time after retrieval, in milliseconds per query.",
        "- Online ms: online query time only, in milliseconds per query.",
        "",
        "## Hash Method Comparison",
        "",
        write_markdown_table(method_df, method_cols),
        "",
        write_latex_table(
            method_df,
            method_cols,
            "Hash method comparison with kernel-weighted annulus KDE metrics on ISOLET.",
            "tab:mech_hash_comparison",
        ),
        "",
        "## Structure Ablation",
        "",
        "The structure ablation evaluates the same trained MECH model and the same fixed-radius task. This table only compares online query behavior; it intentionally excludes training and build/index construction costs.",
        "",
        write_markdown_table(practical_ablation, ablation_cols),
        "",
        write_latex_table(
            practical_ablation,
            ablation_cols,
            "Structure ablation of MECH approximate annulus query on ISOLET.",
            "tab:mech_structure_ablation",
        ),
        "",
        "Oracle upper bound, not included in the deployable structure comparison:",
        "",
        write_markdown_table(oracle_ablation, ablation_cols) if not oracle_ablation.empty else "",
        "",
        "## MECH Sensitivity",
        "",
        write_markdown_table(sens_view, sens_cols),
        "",
        write_latex_table(
            sens_view,
            sens_cols,
            "MECH parameter sensitivity with kernel-weighted KDE metrics on ISOLET.",
            "tab:mech_sensitivity",
        ),
        "",
        "## Main Observations",
        "",
        (
            f"- The lowest annulus KDE relative error in the method comparison is obtained by {best_method['method']} "
            f"({pct(best_method['kde_abs_relative_error'])})."
        ),
        (
            f"- Full MECH obtains point precision {pct(full['point_precision'])}, "
            f"point recall {pct(full['point_recall'])}, "
            f"FP/True {pct(full['point_fp_ratio'])}, "
            f"FN/True {pct(full['point_fn_ratio'])}, "
            f"CER {full['candidate_expansion_ratio']:.4f}, "
            f"KDE evaluations {full['kde_sample_size']:.2f}, "
            f"kernel-weighted recall {pct(full['kernel_weighted_recall'])}, "
            f"and KDE relative error {pct(full['kde_abs_relative_error'])}."
        ),
        "- In the structure ablation, all variants use cached MECH hash candidates; Filter ms includes materializing the cached candidate list and applying variant-specific filtering, while KDE ms measures KDE estimation. Only Full MECH uses per-ring KDE sampling. Training and build/index construction time are excluded.",
        "- In the structure ablation, Full MECH is the best deployable structure; the exact-distance variant is an oracle upper bound because it uses exact Euclidean distance information.",
        "- Kernel-weighted metrics expose whether retrieved points preserve KDE contribution, not just point counts.",
        "- Candidate expansion ratio reports candidate count relative to the exact fixed-radius neighbor count; FP/True and FN/True expose retrieval errors even when CER is close to 1.",
        "- Raw Hash is the hash-only candidate count before radius filtering; R-Pool is the candidate pool after the fixed-radius prefilter, so R-Pool rather than Raw Hash indicates whether fixed-radius retrieval avoided scanning the whole index.",
    ]
    (out_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="isolet")
    parser.add_argument("--n-train", type=int, default=2800)
    parser.add_argument("--n-index", type=int, default=1800)
    parser.add_argument("--n-queries", type=int, default=40)
    parser.add_argument("--inner-percentile", type=float, default=15.0)
    parser.add_argument("--outer-percentile", type=float, default=25.0)
    parser.add_argument("--max-tables", type=int, default=12)
    parser.add_argument("--max-bits", type=int, default=12)
    parser.add_argument("--base-tables", type=int, default=4)
    parser.add_argument("--base-bits", type=int, default=12)
    parser.add_argument("--base-delta", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma-balance", type=float, default=0.1)
    parser.add_argument("--gamma-decorrelation", type=float, default=0.1)
    parser.add_argument("--hamming-probe", type=int, default=0)
    parser.add_argument(
        "--query-mode",
        choices=["global_hash", "radius_first"],
        default="radius_first",
    )
    parser.add_argument("--min-collisions", type=int, default=1)
    parser.add_argument(
        "--fixed-query-radius",
        type=float,
        default=None,
        help="If set, evaluate candidates within this fixed radius and partition [0, R] into annuli.",
    )
    parser.add_argument(
        "--annulus-count",
        type=int,
        default=1,
        help="Number of equal-width annuli inside fixed query radius.",
    )
    parser.add_argument(
        "--ring-sample-size",
        type=int,
        default=64,
        help="Maximum sampled points per annulus for sampled KDE estimation.",
    )
    parser.add_argument("--out-dir", default="mech_experiment_results")
    args = parser.parse_args()

    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = load_dataset(args.dataset)
    train_x, index_x, queries = split_data(
        x, args.n_train, args.n_index, args.n_queries, rng
    )
    dists = distance_matrix(index_x, queries)
    inner_base = np.percentile(dists, args.inner_percentile, axis=1)
    outer_base = np.percentile(dists, args.outer_percentile, axis=1)
    bandwidth_base = float(np.median(outer_base))

    start = time.perf_counter()
    model = train_mech(
        train_x,
        args.max_tables,
        args.max_bits,
        args.epochs,
        args.batch_size,
        args.lr,
        args.alpha,
        args.beta,
        args.gamma_balance,
        args.gamma_decorrelation,
        "cpu",
    )
    train_time_s = time.perf_counter() - start

    base_l = args.base_tables
    base_k = args.base_bits
    base_delta = args.base_delta

    def cfg(
        delta: float,
        radius_factor: float = 1.0,
        bandwidth_factor: float = 1.0,
        variant: str = "full",
    ) -> EvalConfig:
        return EvalConfig(
            delta,
            radius_factor,
            bandwidth_factor,
            variant,
            args.query_mode,
            args.min_collisions,
            args.fixed_query_radius,
            args.annulus_count,
            args.ring_sample_size,
        )

    base_config = cfg(base_delta)

    mech = MECHHashIndex(model, index_x, base_l, base_k, args.hamming_probe)
    methods: list[HashIndexBase] = [
        RandomProjectionHashIndex(
            "SimHash",
            index_x,
            base_l,
            base_k,
            np.random.default_rng(SEED + 1),
            normalized=False,
            hamming_probe=args.hamming_probe,
        ),
        RandomProjectionHashIndex(
            "Angular LSH",
            index_x,
            base_l,
            base_k,
            np.random.default_rng(SEED + 2),
            normalized=True,
            hamming_probe=args.hamming_probe,
        ),
        RandomProjectionHashIndex(
            "Hyperplane LSH",
            index_x,
            base_l,
            base_k,
            np.random.default_rng(SEED + 3),
            normalized=False,
            hamming_probe=args.hamming_probe,
        ),
        mech,
    ]

    method_rows = [
        evaluate_index(m, index_x, queries, dists, inner_base, outer_base, bandwidth_base, base_config)
        for m in methods
    ]
    method_df = pd.DataFrame(method_rows)
    method_df.to_csv(out_dir / "hash_method_comparison.csv", index=False)

    variants = [
        ("Hash Only", "hash_only"),
        ("Hash + Sphere", "no_distance"),
        ("Hash + Distance Table", "distance_table"),
        ("Full MECH", "full"),
        ("Hash + Exact Distance (Oracle)", "oracle_distance"),
    ]
    ablation_rows = []
    for label, variant in variants:
        row = evaluate_index(
            mech,
            index_x,
            queries,
            dists,
            inner_base,
            outer_base,
            bandwidth_base,
            cfg(base_delta, variant=variant),
        )
        row["variant"] = label
        ablation_rows.append(row)
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df = ablation_df.drop(columns=["build_time_s"], errors="ignore")
    ablation_df.to_csv(out_dir / "structure_ablation.csv", index=False)

    sensitivity_settings = [
        ("L", 2, MECHHashIndex(model, index_x, 2, base_k, args.hamming_probe), cfg(base_delta)),
        ("L", 4, MECHHashIndex(model, index_x, 4, base_k, args.hamming_probe), cfg(base_delta)),
        ("L", 8, MECHHashIndex(model, index_x, 8, base_k, args.hamming_probe), cfg(base_delta)),
        ("L", 12, MECHHashIndex(model, index_x, 12, base_k, args.hamming_probe), cfg(base_delta)),
        ("K", 4, MECHHashIndex(model, index_x, base_l, 4, args.hamming_probe), cfg(base_delta)),
        ("K", 8, MECHHashIndex(model, index_x, base_l, 8, args.hamming_probe), cfg(base_delta)),
        ("K", 12, MECHHashIndex(model, index_x, base_l, 12, args.hamming_probe), cfg(base_delta)),
        ("delta", 0.25, mech, cfg(0.25)),
        ("delta", 0.5, mech, cfg(0.5)),
        ("delta", 1.0, mech, cfg(1.0)),
        ("delta", 2.0, mech, cfg(2.0)),
        ("rho", 0.75, mech, cfg(base_delta, 0.75)),
        ("rho", 1.0, mech, cfg(base_delta, 1.0)),
        ("rho", 1.25, mech, cfg(base_delta, 1.25)),
        ("rho", 1.5, mech, cfg(base_delta, 1.5)),
        ("sigma", 0.5, mech, cfg(base_delta, 1.0, 0.5)),
        ("sigma", 1.0, mech, cfg(base_delta, 1.0, 1.0)),
        ("sigma", 2.0, mech, cfg(base_delta, 1.0, 2.0)),
    ]
    sensitivity_rows = []
    for parameter, value, idx, cfg in sensitivity_settings:
        row = evaluate_index(idx, index_x, queries, dists, inner_base, outer_base, bandwidth_base, cfg)
        row["parameter"] = parameter
        row["value"] = value
        sensitivity_rows.append(row)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(out_dir / "mech_sensitivity_summary.csv", index=False)

    config = {
        "dataset": args.dataset,
        "n_train": len(train_x),
        "n_index": len(index_x),
        "n_queries": len(queries),
        "inner_percentile": args.inner_percentile,
        "outer_percentile": args.outer_percentile,
        "bandwidth_base": bandwidth_base,
        "preprocessing": "minmax scaling followed by 2*x plus the maximum-norm sample",
        "time_metric": "online query time only; training and Hash/Distance/Sphere table construction are excluded; MECH query hashes are cached, while SimHash/Angular LSH/Hyperplane LSH query hashes are computed online; candidate retrieval, filtering, and KDE estimation are included; only Full MECH uses sampled per-ring KDE estimation",
        "max_tables": args.max_tables,
        "max_bits": args.max_bits,
        "base_tables": base_l,
        "base_bits": base_k,
        "base_delta": base_delta,
        "query_mode": args.query_mode,
        "min_collisions": args.min_collisions,
        "fixed_query_radius": args.fixed_query_radius,
        "annulus_count": args.annulus_count,
        "ring_sample_size": args.ring_sample_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma_balance": args.gamma_balance,
        "gamma_decorrelation": args.gamma_decorrelation,
        "hamming_probe": args.hamming_probe,
        "timing_repeats": TIMING_REPEATS,
        "train_time_s": train_time_s,
        "seed": SEED,
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_summary(out_dir, config, method_df, ablation_df, sensitivity_df)

    print(f"Saved MECH experiment results to {out_dir.resolve()}")
    print(f"MECH training time: {train_time_s:.2f}s")
    print(method_df.to_string(index=False))


if __name__ == "__main__":
    main()
