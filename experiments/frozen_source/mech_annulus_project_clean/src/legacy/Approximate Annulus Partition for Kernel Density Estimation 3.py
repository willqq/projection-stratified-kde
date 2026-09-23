import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from collections import defaultdict
import pickle
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import scipy.io as scio
from scipy.io import arff
import datetime

# 设置随机种子
seed = 1024
# seed = 1021
torch.manual_seed(seed)  # 设置CPU的随机种子
np.random.seed(seed)     # 设置NumPy的随机种子
random.seed(seed)        # 设置Python内置的随机种子
delta = 1
num_tables = 2
num_planes = 10
# 假设有不同的k值
k_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
recall_values = []
precision_values = []
candidate_accuracy_values = []


def min_max_scaler(original_data):
    scaler = MinMaxScaler()
    scaler.fit(original_data)
    normalized_data = pd.DataFrame(scaler.transform(original_data)).fillna(0).values
    return normalized_data


import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
from collections import defaultdict

import numpy as np
from typing import Dict, List, Tuple


class AngularLSH:
    def __init__(self, dim: int, num_tables: int = 10, hash_length: int = 16):
        """
        Args:
            dim: 输入向量维度
            num_tables: 哈希表数量 (L)
            hash_length: 每个哈希表的哈希位数 (k)
        """
        self.dim = dim
        self.L = num_tables
        self.k = hash_length

        self.table = defaultdict(list)
        # 生成随机投影矩阵列表（每个矩阵 shape=[k, dim]）
        self.projections = [self._generate_projection_matrix() for _ in range(num_tables)]

    def _generate_projection_matrix(self) -> np.ndarray:
        """生成归一化的随机投影矩阵（超平面法向量）"""
        matrix = np.random.randn(self.k, self.dim)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """L2归一化（余弦相似度需要单位向量）"""
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    def set_num_tables(self, num_tables):
        self.L = num_tables

    def hash(self, x: np.ndarray,indexs=[]) -> List[int]:
        """计算哈希值（返回每个表的k-bit二进制编码）"""
        # x_norm = self._normalize(x)
        hashes = defaultdict()
        if len(indexs)!=0:

            for i, proj in enumerate(self.projections[ :self.L ]):
                # if i in self.table.keys():
                #     continue
                hash=defaultdict(list)
                for index in indexs:
                    point=x[index]
                    # 投影并二值化 (sign(proj·x) → 1/0)
                    bits = (np.dot(proj, point) > 0).astype(int)
                    # 将k-bit二进制转换为整数（方便存储和比较）
                    hash_val = int(''.join(map(str, bits)), 2)
                    hash[hash_val].append(index)
                hashes[i]=hash
        else:
            for i, proj in enumerate(self.projections[:self.L]):
                point = x
                # 投影并二值化 (sign(proj·x) → 1/0)
                bits = (np.dot(proj, point) > 0).astype(int)
                # 将k-bit二进制转换为整数（方便存储和比较）
                hash_val = int(''.join(map(str, bits)), 2)
                hashes[i] = hash_val

        return hashes

    def insert(self, data,id):
        """构建哈希索引 {id: [hash_table1, ...]}"""
        self.table=self.hash(data,id)

    def query(
            self,
            query_vec: np.ndarray,
            top_n: int = 5,
            threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        查询最近邻（基于哈希匹配数估计相似度）
        Returns:
            List[Tuple[id, estimated_similarity]]
        """
        query_hash = self.hash(query_vec)
        scores = []
        for table_index,hash in query_hash.items():

             scores.extend(  self.table[table_index].get(hash,[]) )  # 获取所有表中匹配的哈希值
            # for id_, data_hash in table.items():
            #     # 计算所有表中匹配的哈希位数
            #     for h1, h2 in zip(query_hash, data_hash):
            #         print(h1, h2 )
            #     matched = sum(h1 == h2 for h1, h2 in zip(query_hash, data_hash))
            #     print(matched)
            #     # 估计相似度：sim ≈ matched / L
            #     if  matched >0:
            #         scores.append(id_)

        return scores
class CrossPolytopeLSH:
    def __init__(self, dim, L=10):
        """
        dim: 向量维度
        k: 每个哈希表的哈希函数数量(通常为1)
        L: 哈希表数量
        """
        self.dim = dim
        self.L = L

        # 生成L个随机旋转矩阵
        self.rotations = [self._generate_rotation(dim) for _ in range(L)]
    def set_num_tables(self, num_tables):
        self.L = num_tables
        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
    def _generate_rotation(self, dim):
        """生成随机旋转矩阵(改进版)"""
        # 使用Hadamard变换+对角矩阵加速
        H = np.random.choice([-1, 1], size=(dim, dim))
        D = np.diag(np.random.randn(dim))
        return D @ H

    def _hash_func(self, vec, rotation):
        """单个哈希函数计算"""
        rotated = rotation @ vec
        max_idx = np.argmax(np.abs(rotated))
        sign = rotated[max_idx] >= 0
        return (int(max_idx), bool(sign))

    def insert(self, vec, id):
        """插入向量到所有哈希表"""
        vec = vec / np.linalg.norm(vec)
        for i in range(self.L):
            key = self._hash_func(vec, self.rotations[i])
            self.hash_tables[i][key].append(id)

    def query(self, vec, top_n=10):
        """查询相似向量"""
        vec = vec / np.linalg.norm(vec)
        candidates = set()

        # 查询所有哈希表
        for i in range(self.L):
            key = self._hash_func(vec, self.rotations[i])
            candidates.update(self.hash_tables[i].get(key, []))

        return list(candidates)

class SimHashMultiTable:
    def __init__(self, dim: int, hash_length: int = 64, strategy: str = "partition"):
        """
        SimHash 多表实现
        :param dim: 数据维度
        :param hash_length: 哈希码长度（默认64-bit）
        :param num_tables: 表数量（分块数或多哈希组数）
        :param strategy: 多表策略，可选 "partition"（分块）或 "multi_hash"（多哈希函数）
        """
        self.dim = dim
        self.hash_length = hash_length
        self.num_tables = num_tables
        self.strategy = strategy

        # 初始化投影矩阵和哈希表
        if strategy == "partition":
            # 分块多表：单组投影，逻辑分块
            self.projections = np.random.randn(hash_length, dim)
        elif strategy == "multi_hash":
            # 多哈希函数组合：多组独立投影
            self.projections_list = [np.random.randn(hash_length, dim) for _ in range(num_tables)]
            self.tables = [defaultdict(list) for _ in range(num_tables)]
        else:
            raise ValueError("策略必须是 'partition' 或 'multi_hash'")

    def _simhash(self, vec: np.ndarray, projections: np.ndarray) -> Tuple[int]:
        """生成SimHash二进制码"""
        return tuple(1 if np.dot(r, vec) >= 0 else 0 for r in projections)
    def set_num_tables(self, num_tables):
        self.num_tables = num_tables
        self.tables = [defaultdict(list) for _ in range(num_tables)]
    def insert(self, vec, id):

        """插入数据"""
        if self.strategy == "partition":
            full_hash = self._simhash(vec, self.projections)
            block_size = self.hash_length // self.num_tables
            for i in range(self.num_tables):
                block = full_hash[i * block_size: (i + 1) * block_size]
                self.tables[i][block].append(id)
        else:  # multi_hash
            for i in range(self.num_tables):
                hash_code = self._simhash(vec, self.projections_list[i])
                self.tables[i][hash_code].append(id)

    def query(self, query_vec: np.ndarray, top_n: int = 5) -> List[int]:
        """查询最近邻"""
        candidates = set()

        if self.strategy == "partition":
            full_hash = self._simhash(query_vec, self.projections)
            block_size = self.hash_length // self.num_tables
            for i in range(self.num_tables):
                block = full_hash[i * block_size: (i + 1) * block_size]
                if block in self.tables[i]:
                    candidates.update(self.tables[i][block])
                # for hash in  self.tables[i].keys():
                #     # print(hash)
                #     dis=sum(c1 != c2 for c1, c2 in zip(hash, block))
                #     if dis<=2:
                #         candidates.update(self.tables[i][hash])

        else:  # multi_hash
            for i in range(self.num_tables):
                query_hash = self._simhash(query_vec, self.projections_list[i])
                if query_hash in self.tables[i]:
                    candidates.update(self.tables[i][query_hash])

        # 按余弦相似度排序（需外部实现）
        return candidates

    # def _rank_candidates(self, query_vec: np.ndarray, candidates: List[int]) -> List[int]:
    #     """按余弦相似度排序候选（需根据实际数据实现）"""
    #     # 示例：假设有全局数据列表 data = [vec1, vec2, ...]
    #     similarities = [
    #         (id, np.dot(query_vec, data[id]) / (np.linalg.norm(query_vec) * np.linalg.norm(data[id])))
    #         for id in candidates
    #     ]
    #     return [id for id, _ in sorted(similarities, key=lambda x: x[1], reverse=True)]
def get_cifar_10_data():
    print("dataset: cifar-10-batches")
    data = []
    #提出数据
    for i in range(1, 2):
        file = 'data/cifar-10-batches-py/data_batch_' + str(i)
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            #选择data数据
            data.extend(dict[b'data'])

    #归一化
    data = min_max_scaler(data)
    return data

def get_isolet_data():
    print("dataset: isolet")
    data = pd.read_csv('data/isolet/isolet1+2+3+4.data', sep=',')

    # 归一化
    data = min_max_scaler(data)
    return data

def get_Amazon_data():
    print("dataset: Amazon")
    # 读取 ARFF 数据
    data, meta = arff.loadarff("data/Amazon_initial_50_30_10000/Amazon_initial_50_30_10000.arff")
    data = pd.DataFrame(data).drop(columns=['class_duplicate'])

    # 归一化
    data = min_max_scaler(data)
    return data

def get_cifar10_Gist512_data():
    print("dataset: cifar10_Gist512")
    data = scio.loadmat('data/cifar10-Gist512/Cifar10-Gist512.mat')['X']

    # 归一化
    data = min_max_scaler(data)
    return data


def translate_points_to_radius(q, r, data):
    # 计算查询点到原点的距离
    distance_to_origin = np.linalg.norm(q)

    # 如果查询点到原点的距离小于查询半径 r，则进行平移
    if distance_to_origin < r:
        # 计算平移向量 t
        translation_vector = (r - distance_to_origin) * q / distance_to_origin
        # 平移点集中的所有点
        data_translated = data + translation_vector
        # 平移查询点 q
        q_translated = q + translation_vector
        return data_translated, q_translated
    else:
        # 如果查询点已经在有效范围内，返回原始数据
        return data, q


def calculate_ring_parameters(query_point, data, percentile_10=10, percentile_30=30, percentile_50=50):
    """
    计算查询点与数据集中所有点的第10%和第30%分位点的距离
    :param query_point: 查询点（向量）
    :param data: 数据集（每行一个数据点）
    :param percentile_10: 第10%分位点
    :param percentile_30: 第30%分位点
    :return: 第10%和第30%分位点的距离（r和r_z）
    """
    # 计算查询点与数据集中所有点的欧几里得距离
    distances = np.linalg.norm(data - query_point, axis=1)

    # 计算第10%和第30%分位点的距离
    r1 = np.percentile(distances, percentile_10)
    r2 = np.percentile(distances, percentile_30)
    r3 = np.percentile(distances, percentile_50)

    return r1, r2, r3


# ------------------------ Step 1: Sphere Table (Spherical Layer) Construction ------------------------

def build_sphere_table(points, delta=1.0):
    """
    Build Sphere Table based on spherical layers (divided by delta).
    :param points: Data matrix (n_samples, d)
    :param delta: Interval to divide the spherical layers
    :return: Sphere Table where each layer contains points within a specific range
    基于球面层构建球体表（按delta划分）。
    : 参数data：数据矩阵（n_samples, d）
    : 参数delta：划分球面层的间隔
    ：return：球体表，每层包含一个特定范围内的点
    """
    sphere_table = {}
    for point_id, point in enumerate(points):
        norm = np.linalg.norm(point)
        layer = round(norm / delta)
        if layer not in sphere_table:
            sphere_table[layer] = []
        sphere_table[layer].append(point_id)

    return sphere_table


# ------------------------ Step 2: Hash Tables ------------------------

# 生成随机超平面用于构建 LSH
def generate_random_hyperplanes(dimension, num_planes):
    return np.random.randn(num_planes, dimension)


# 单个点的哈希值计算
def hash_point(point, hyperplanes):
    return ''.join(['0' if np.dot(h, point) <= 0 else '1' for h in hyperplanes])


# 构建 Hash Tables
def build_hash_tables(points, filtered_candidates, query_point,num_tables, num_planes, dimension):
    global hypers
    hash_tables = []
    hyperplane_sets = []
    query_hash_table=[]
    hyperplane_sets.extend(hypers[:num_tables])
    for i in range(num_tables):
        hyperplanes =hyperplane_sets[i]  # 每个表使用不同的随机超平面
        # hyperplane_sets.append(hyperplanes)  # 保存随机超平面
        hash_table = defaultdict(list)

        for point_id in filtered_candidates:
            point = points[point_id]
            hash_value = hash_point(point, hyperplanes)  # 计算点的哈希值
            hash_table[hash_value].append(point_id)  # 点的索引存入对应的哈希桶
        # 计算查询点的哈希值
        query_hash_value = hash_point(query_point, hyperplanes)
        query_hash_table.append(query_hash_value)
        hash_tables.append(hash_table)

    return hash_tables, hyperplane_sets,query_hash_table


# 构建 Distance Tables
def build_distance_tables(hash_tables):
    distance_tables = []

    for table in hash_tables:
        distance_table = {}

        for hash_value in table.keys():
            # 计算与当前哈希值的所有可能汉明距离
            distances = defaultdict(list)
            for other_hash_value in table.keys():
                hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash_value, other_hash_value))
                distances[hamming_distance].append(other_hash_value)

            distance_table[hash_value] = distances

        distance_tables.append(distance_table)

    return distance_tables


# ------------------------ Step 3: Hyperplane LSH (for Angular Query) ------------------------

def query_candidate_points(query_hash_value, hash_tables, distance_tables, max_hamming_distance=1,use_hamming_distance=False):
    """根据哈希值查询候选点"""
    candidate_points = []

    # 从每个哈希表中查询哈希值相同或汉明距离在阈值以内的点
    if query_hash_value in hash_tables:
        candidate_points.extend(hash_tables[query_hash_value])  # 查询哈希值相同的点
    # print(len(candidate_points), "candidate_points")
    # 通过汉明距离查询可能的点
    if use_hamming_distance:
        for hamming_distance in range(1, max_hamming_distance + 1):
            if query_hash_value in distance_tables:
                for candidate_hash in distance_tables[query_hash_value].get(hamming_distance, []):
                    candidate_points.extend(distance_tables[candidate_hash])  # 查询与给定哈希值汉明距离为 i 的点
    print(len(candidate_points), "candidate_points")

    return list(candidate_points)


def max_angle(query_point, radius):
    """计算给定查询点和半径的最大角度"""
    norm_q = np.linalg.norm(query_point)  # 计算查询点的范数 ||q||
    if norm_q == 0:
        return 0  # 如果查询点的范数为零，角度为零

    # 使用反正弦函数（arcsin）计算最大角度
    theta_max = np.arcsin(radius / norm_q)
    return theta_max


def filter_candidates_by_angle(points, query_point, radius, sphere_table, max_angle_value):
    """通过角度过滤候选点"""
    filtered_candidates = []

    q_norm = np.linalg.norm(query_point)
    fs = round((q_norm - radius) / delta)
    ls = round((q_norm + radius) / delta)

    for layer in range(fs, ls + 1):

        if layer not in sphere_table:
            continue

        for candidate_index in sphere_table[layer]:
            candidate_point = points[candidate_index]
            # 计算查询点和候选点之间的角度
            dot_product = np.dot(query_point, candidate_point)  # 内积
            norm_candidate = np.linalg.norm(candidate_point)  # 候选点的范数 ||p||

            if norm_candidate == 0:
                angle = 0  # 如果候选点的范数为零，角度为零
            else:
                cos_angle = dot_product / (np.linalg.norm(query_point) * norm_candidate)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 计算角度，并进行夹紧防止数值不稳定

            #如果角度小于最大角度，保留该候选点
            if angle <= max_angle_value:
                filtered_candidates.append(candidate_index)
            # filtered_candidates.append(candidate_index)

    return filtered_candidates


def filter_candidates_by_distance(candidates, points, query_point, radius):
    """通过欧式距离过滤候选点"""
    filtered_candidates = []
    l1=np.linalg.norm(query_point)
    for candidate_index in candidates:
        candidate_point = points[candidate_index]  # 假设 points 是预处理过的所有数据点
        distance = np.linalg.norm(candidate_point)  # 计算欧式距离
        if distance <= radius+l1:
            filtered_candidates.append(candidate_index)
    return filtered_candidates

def query_with_max_angle(points, query_point, radius, sphere_table, k, num_planes, dim, max_hamming_distance=1,strategy="HyperplaneLSH"):
    """执行查询，考虑最大角度限制"""
    candidate_points = []
    # 步骤 1: 计算查询点的最大角度
    if strategy == "HyperplaneLSH" or strategy == "HyperplaneLSH+HanMing":
        num_planes=int(100/k)
        candidate_points = []
        if strategy == "HyperplaneLSH":
            use_hamming_distance=False
            dist_thre = 2  # max(1, int(k * p * 0.3))
        if strategy == "HyperplaneLSH+HanMing":
            use_hamming_distance=True
            dist_thre = 1  # max(1, int(k * p *
        max_angle_value = max_angle(query_point, radius)
        p = 1 - max_angle_value / np.pi  # 单层匹配概率
        print("p",p)

        # 步骤 4: 过滤候选点（根据角度）
        filtered_candidates = filter_candidates_by_angle(points, query_point, radius, sphere_table, max_angle_value)
        print(len(filtered_candidates), k, num_planes, "filtered_candidates" + strategy)
        if use_hamming_distance :
            filtered_candidates = filter_candidates_by_distance(filtered_candidates, points, query_point, radius)
        hash_tables, hyperplane_sets,query_hash_table = build_hash_tables(points, filtered_candidates,query_point, k, num_planes, dim)
        distance_tables = build_distance_tables(hash_tables)
        print( len(filtered_candidates), k, num_planes, "filtered_candidates" + strategy)
        for i, hyperplane_set in enumerate(hyperplane_sets):
            query_hash_value = query_hash_table[i]#hash_point(query_point, hyperplane_set)  # 计算查询点在第 i 个哈希表中的哈希值
            print(query_hash_value,hash_point(query_point, hyperplane_set))
            candidate_points.extend(
                query_candidate_points(query_hash_value, hash_tables[i], distance_tables[i], max_hamming_distance,use_hamming_distance))
        print(i,len(candidate_points)/len(set(candidate_points)), p, k, num_planes, "candidate_points")
        dicts = {}
        for key in candidate_points:
            dicts[key] = dicts.get(key, 0) + 1
        # print(dicts)
        candidate_points=[]
        for key in dicts:
            if strategy == "HyperplaneLSH+HanMing":
                if dicts[key] >= dist_thre:
                    candidate_points.extend([key] * dicts[key])
            else:
                if dicts[key] >= dist_thre:
                    candidate_points.extend([key] * dicts[key])


        print("dist_thre", dist_thre)
        print("candidate_points",len(candidate_points))
    if strategy == "SimHash":
        global  simhash
        candidate_points = []
        use_hamming_distance = False
        max_angle_value = max_angle(query_point, radius)
        p = 1 - max_angle_value / np.pi  # 单层匹配概率
        print("p", p)
        if use_hamming_distance:
            dist_thre = 1  # max(1, int(k * p * 0.3))
        else:
            dist_thre = 2  # max(1, int(k * p * 0.3))
        max_hamming_distance = 1
        # 步骤 4: 过滤候选点（根据角度）
        filtered_candidates = filter_candidates_by_angle(points, query_point, radius, sphere_table, max_angle_value)
        # filtered_candidates = filter_candidates_by_distance(filtered_candidates, points, query_point, radius)
        simhash.set_num_tables( k)
        for i in filtered_candidates:

            simhash.insert(points[i], i)
        candidate_points=simhash.query(query_point)  # 查询点的哈希值
        print("candidate_points", len(candidate_points))
    if strategy == "CrossPolytopeLSH":
        global crosspolytope
        candidate_points = []
        use_hamming_distance = False
        max_angle_value = max_angle(query_point, radius)
        p = 1 - max_angle_value / np.pi  # 单层匹配概率
        print("p", p)
        if use_hamming_distance:
            dist_thre = 1  # max(1, int(k * p * 0.3))
        else:
            dist_thre = 2  # max(1, int(k * p * 0.3))
        max_hamming_distance = 1
        # 步骤 4: 过滤候选点（根据角度）
        filtered_candidates = filter_candidates_by_angle(points, query_point, radius, sphere_table, max_angle_value)
        # filtered_candidates = filter_candidates_by_distance(filtered_candidates, points, query_point, radius)
        crosspolytope.set_num_tables(k)
        for i in filtered_candidates:
            crosspolytope.insert(points[i], i)
        candidate_points = crosspolytope.query(query_point)  # 查询点的哈希值
        # print("candidate_points", len(candidate_points))
    if strategy == "AngularLSH":
        global angularlsh

        candidate_points = []
        use_hamming_distance = False
        max_angle_value = max_angle(query_point, radius)
        p = 1 - max_angle_value / np.pi  # 单层匹配概率
        print("p", p)
        if use_hamming_distance:
            dist_thre = 1  # max(1, int(k * p * 0.3))
        else:
            dist_thre = 2  # max(1, int(k * p * 0.3))
        max_hamming_distance = 1
        # 步骤 4: 过滤候选点（根据角度）
        filtered_candidates = filter_candidates_by_angle(points, query_point, radius, sphere_table, max_angle_value)
        # filtered_candidates = filter_candidates_by_distance(filtered_candidates, points, query_point, radius)
        angularlsh.set_num_tables(k)

        angularlsh.insert(points, filtered_candidates)
        candidate_points = angularlsh.query(query_point,threshold=1-p)  # 查询点的哈希值

    return candidate_points



# ------------------------ Step 3: Approximate Annulus Partition ------------------------

def approximate_ring_query_multiple_radii(points, query_point, radii, sphere_table, k, num_planes, dim, max_hamming_distance=1,strategy="HyperplaneLSH"):
    """
    执行多个半径的近似环查询。
    radii: 包含多个半径 [r1, r2, r3, ..., rn]
    """
    # 初始化结果集合
    candidate_sets = []
    candidate_list=[]
    # 步骤 1: 查询每个半径的近邻集合
    for radius in radii:
        candidate_points = query_with_max_angle(points, query_point, radius, sphere_table, k, num_planes, dim, max_hamming_distance,strategy=strategy)
        candidate_sets.append(candidate_points)  # 将结果存入集合

    # 步骤 2: 计算每对相邻半径的差集
    ring_queries = [list(set(candidate_sets[0]))]
    for i in range(len(radii) - 1):
        r1_set = candidate_sets[i]
        r2_set = candidate_sets[i + 1]
        ring_queries.append(list(set(r2_set) - set(r1_set)))  # 计算 C(r2) - C(r1)
    candidate_list.extend(candidate_sets[0])  # 将第一个集合添加到候选列表
    candidate_list.extend(candidate_sets[1])  # 将第一个集合添加到候选列表
    return ring_queries,candidate_list  # 返回所有的近似环查询结果


# ------------------------ Step 4: Kernel Density Estimation ------------------------

# 高斯核函数
def gaussian_kernel(q, p, bandwidth=1.0):
    """计算高斯核函数"""
    return np.exp(-np.linalg.norm(q - p) ** 2 / (2 * bandwidth ** 2))


def compute_density(query_point, candidates, points, bandwidth=1.0):
    """计算给定点 q 在候选点集中的核密度估计"""
    density = 0.0
    for idx in candidates:
        p = points[idx]
        density += gaussian_kernel(query_point, p, bandwidth)
    return density / len(candidates)  # 核密度的均值

def z_alpha_half(alpha):
    from scipy.stats import norm
    return norm.ppf(1 - alpha / 2)

def annulus_monte_carlo_sample(query_point, candidates, points, mbegin=100, epsilon=0.5, alpha=0.5, beta=2.0, bandwidth=1.0):
    """使用蒙特卡洛方法对指定环内的点进行抽样计算核密度"""

    n_i = len(candidates)  # 数据集A_i中的点的数量

    # 如果抽样数量足够，直接计算核密度估计
    if beta * mbegin >= n_i:
        # 直接计算核密度估计的函数，这里假设有一个函数可以计算
        return compute_density(query_point, candidates, points, bandwidth), candidates

    m_rest = mbegin  # 初始抽样数量
    while True:
        # 从A_i中均匀随机选择m_rest个点
        print(len(candidates), m_rest)
        sampled_points = np.random.choice(candidates, m_rest, replace=False)

        # 计算每个点与查询点q的核函数值
        values = [gaussian_kernel(query_point, points[idx], bandwidth) for idx in sampled_points]

        # 计算当前样本的期望值和方差
        m = len(values)
        mu_hat = np.mean(values)
        sigma_hat = np.var(values)

        # 根据公式计算所需的理想样本数
        m_need = int((z_alpha_half(alpha)**2 * sigma_hat) / (epsilon**2 * mu_hat**2))

        print(m_need, m, m_rest)

        # 如果当前样本数已足够，退出循环
        if m_need <= m:
            break

        # 更新剩余需要的样本数
        m_rest = m_need - m

    print(len(sampled_points))

    return mu_hat, sampled_points


def annulus_density_estimator(query_point, ring_queries, points):
    """对整个近似环划分进行核密度估计"""
    total_density = 0.0
    total_count = 0
    all_sampled_points = []

    for ring in ring_queries:
        if len(ring) > 0:
            # 对每一层的候选点进行核密度估计
            density, sampled_points = annulus_monte_carlo_sample(query_point, ring, points)
            total_density += density * len(ring)  # 每层密度的加权和
            total_count += len(ring)  # 累计总点数
            all_sampled_points.append(sampled_points)

            # 计算总的核密度估计
    return total_density / total_count if total_count > 0 else 0.0, all_sampled_points


# ------------------------ traditional_density_estimator ------------------------

def traditional_density_estimator(query_point, points, bandwidth=1.0,radius=10e9):
    """传统的核密度估计，计算所有点的核密度"""
    density = 0.0
    n=0
    for p in points:
        distances = np.linalg.norm(p - query_point)
        if distances<radius:
            n = n+1
            density += gaussian_kernel(query_point, p, bandwidth)
    return density / n  # 核密度的均值
def density_estimator(query_point, points,indexes, bandwidth=1.0,radius=10e9,type="mcmc",huan_num=30,sample_num=1000 ):
    #计算精确核密度
    res=defaultdict(list)
    n = 0
    density=0
    for p in range(len(points)):
        distances = np.linalg.norm(points[p] - query_point)
        if distances < radius:
            n = n + 1
            density += gaussian_kernel(query_point, points[p], bandwidth)
    density=density/n
    n = 0
    appro_density1 = 0
    print(indexes)
    for p  in  indexes:
        p=int( p)
        print(p)
        distances = np.linalg.norm(points[p] - query_point)
        if distances < radius:
            n = n + 1
            appro_density1 += gaussian_kernel(query_point, points[p], bandwidth)
    appro_density1 = appro_density1 / n
    print("appro_density1",appro_density1,density)
    #计算抽样核密度
    appro_density = 0.0
    radiuses = np.linspace(0, radius, huan_num)
    for type_index,i in enumerate([range(len(points)),indexes]):
        acc_huan_n = 0
        acc_huan = defaultdict(list)
        for p in i:
            p=int(p)
            distances = np.linalg.norm(points[p] - query_point)
            if distances < radius:
                acc_huan_n=acc_huan_n+1
                for huan, huan_radius in enumerate(range(len(radiuses) - 1)):
                    if distances >= radiuses[huan] and distances < radiuses[huan + 1]:
                        acc_huan[radiuses[huan_radius]].append(p)


        for huan_index in acc_huan.keys():
            print("huan_index",huan_index, len(acc_huan[huan_index]))
        for sample_num1 in range(20,sample_num+1,20):
            acc_density=0
            n=0
            for huan_index in acc_huan.keys():
                # print(huan_index,len(acc_huan[huan_index]))
                num=int(len(acc_huan[huan_index])/acc_huan_n*sample_num1)
                # print(num)
                huan_sams=np.random.choice(acc_huan[huan_index], max(num,1), replace=True)
                n = n +  len(acc_huan[huan_index])
                for sam in huan_sams:

                    acc_density += len(acc_huan[huan_index])*gaussian_kernel(query_point, points[sam], bandwidth)/len(huan_sams)
            # print("acc_density",acc_density)
            acc_density = acc_density / n
            if type_index==0:
                res["sample_num"].append(sample_num1)
                res["acc_density"].append(abs(density-acc_density))

            else:
                res["appro_density"].append(abs(density-acc_density))
            print("sample_num1",n,type_index,density,sample_num1,"abs(density-acc_density)",abs(density-acc_density),acc_density,appro_density1)
    #mcmc
    ps=[]
    for p in indexes:
        p=int(p)
        distances = np.linalg.norm(points[p] - query_point)
        if distances < radius:
            ps.append(p)
    for sample_num1 in range(20, sample_num+1, 20):
        appro_density = 0
        # print(indexes)
        sams = np.random.choice(ps, sample_num1, replace=True)
        for sam in sams:
            distances = np.linalg.norm( points[sam] - query_point)
            # print("distances",distances)
            appro_density += gaussian_kernel(query_point, points[sam], bandwidth)
            # print("appro_density",appro_density)
        appro_density=appro_density/len(sams)

        print("sample_num11",sample_num1,density,"abs(density-appro_density)",abs(density-appro_density),appro_density,appro_density1,len(ps))

        res["mcmc_density"].append(abs(density-appro_density))

    return res # 核密度的均值

# ------------------------ 实验指标 ------------------------

def get_true_neighbors(query_point, points, radius):
    """
    计算与查询点距离小于给定半径的真实邻居点数量
    """
    distances = np.linalg.norm(points - query_point, axis=1)
    true_neighbors = set(np.where(distances <= radius)[0])
    return true_neighbors

def compute_recall(true_neighbors, retrieved_neighbors):
    """
    计算召回率
    """
    print(len(set(true_neighbors) & set(retrieved_neighbors)),  len(true_neighbors))
    return len(set(true_neighbors) & set(retrieved_neighbors))  / len(true_neighbors) if len(true_neighbors) > 0 else 0


def compute_precision(true_neighbors, retrieved_neighbors):
    """
    计算准确率
    """
    num = 0
    for i in retrieved_neighbors:
        if i in true_neighbors:
            num+=1
    print(num/len(retrieved_neighbors),"num/len(retrieved_neighbors)")
    print(len(set(true_neighbors) & set(retrieved_neighbors)),  len(retrieved_neighbors))
    return num/len(retrieved_neighbors)#len(set(true_neighbors) & set(retrieved_neighbors))  / len(set(retrieved_neighbors)) if len(true_neighbors) > 0 else 0




def get_candidate_accuracy(query_point, candidate_set, true_radius):
    """
    计算候选集准确率，衡量在候选集中的点有多少是真正与查询点相关的
    :param query_point: 查询点
    :param candidate_set: 近似查询得到的候选集
    :param true_radius: 判断相关数据的精确距离阈值
    :return: 候选集准确率
    """
    candidate = []
    for idx in candidate_set:
        p = points[idx]
        candidate.append(p)

    if len(candidate) == 0:
        return 0

    candidate = np.array(candidate)

    # 计算每个候选点与查询点的欧几里得距离
    distances = np.linalg.norm(candidate - query_point, axis=1)

    # 判断哪些点是与查询点相关的，假设相关数据是指距离小于给定真实半径的点
    relevant_data_count = np.sum(distances <= true_radius)

    # 候选集的总数据数量
    total_candidate_count = len(candidate_set)

    # 计算候选集准确率
    candidate_accuracy = relevant_data_count / total_candidate_count if total_candidate_count > 0 else 0
    return candidate_accuracy

def plot_results(x, recall, precision, candidate_accuracy=None,title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(x, recall, marker='o', label="Recall", color='blue')
    plt.plot(x, precision, marker='s', label="Precision", color='green')
    # plt.plot(x, candidate_accuracy, marker='^', label="Candidate Accuracy", color='red')
    plt.xlabel("k (Hash Code Length)")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.show()

# ------------------------ Example Usage ------------------------

# Generate random data (1000 samples, 10 dimensions)

# points = np.random.rand(1000, 10)  # 1000个128维的点
if __name__ == '__main__':

    # points = get_cifar_10_data()
    # points = get_isolet_data()
    # points = get_Amazon_data()
    cls = "cifar10"
    mata_data = {"cifar10": 5, "isolet": 3, "Amazon": 2, "cifar10_Gist512": 5}
    if cls == "cifar10":
        num_planes = mata_data[cls]
        points = get_cifar_10_data()
    elif cls == "isolet":
        num_planes = mata_data[cls]
        points = get_isolet_data()
    elif cls == "Amazon":
        num_planes = mata_data[cls]
        points = get_Amazon_data()
    elif cls == "cifar10_Gist512":
        num_planes = mata_data[cls]
        points = get_cifar10_Gist512_data()
    dim = points.shape[1]
    print(points.shape)

    # query_point = np.random.rand(dim)
    q_point = np.random.choice(points.shape[0])
    query_point = points[q_point]
    r1, r2, r3 = calculate_ring_parameters(query_point, points, 10, 30, 50)
    radii = [r1, r2]
    radius = r2
    print("r1:", r1, "r2:", r2, "r3:", r3)
    print("---------------")
    hypers = []
    points, query_point = translate_points_to_radius(query_point, radii[-1], points)

    for _ in range(max(k_values)):
        hyperplanes = generate_random_hyperplanes(dim, num_planes)
        hypers.append(hyperplanes)
    simhash = SimHashMultiTable(dim, hash_length=100, strategy="partition")
    # crosspolytope=CrossPolytopeLSH(dim, max(k_values))
    angularlsh = AngularLSH(dim, max(k_values), num_planes - 1)
    strates=["HyperplaneLSH+HanMing","HyperplaneLSH","SimHash","AngularLSH",]#"CrossPolytopeLSH"
    R = {}
    P = {}
    F1 = {}
    for i in range(len(strates)):
        strategy = strates[i]

        #传统核密度估计

        starttime = datetime.datetime.now()


        usetime = 0

        # 第一步：构建数据结构
        sphere_table = build_sphere_table(points, delta)
        recall_values= []
        precision_values=[]
        f1_values=[]
        candidate_set=[]
        for k in k_values:

            starttime_k = datetime.datetime.now()

            ring_queries,candidate_list = approximate_ring_query_multiple_radii(points, query_point, radii, sphere_table, k, num_planes, dim, max_hamming_distance=1,strategy=strategy)
            estimated_density, sampled_points = annulus_density_estimator(query_point, ring_queries, points)
            print("num_tables:", k, ", Estimated density:", estimated_density)
            endtime_k = datetime.datetime.now()
            print("num_tables:", k, ", time:", (endtime_k - starttime_k).seconds)
            usetime += (endtime_k - starttime_k).seconds

            candidate_set = np.concatenate(ring_queries)
            sampled_points = np.concatenate(sampled_points)

            true_neighbors = get_true_neighbors(query_point, points, radius)
            retrieved_neighbors = len(candidate_set)
            sampled_points_number = len(sampled_points)
            total_retrieved_neighbors = len(points)

            recall = compute_recall(true_neighbors, candidate_set)
            precision = compute_precision(true_neighbors, candidate_list)

            # candidate_accuracy = get_candidate_accuracy(query_point, candidate_set, radius)

            recall_values.append(recall)
            precision_values.append(precision)
            # candidate_accuracy_values.append(candidate_accuracy)
            f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
            f1_values.append(f1)
            traditional_density = traditional_density_estimator(query_point, points, bandwidth=1.0,radius=radius)

            endtime = datetime.datetime.now()


            # if k==k_values[2] and strategy=="HyperplaneLSH+HanMing":
            #     print("---------------")
            #     density=density_estimator(query_point, points,set(candidate_set),radius=radius,huan_num=10,sample_num=500)
            #     df = pd.DataFrame(density)
            #     df.to_csv("output_{}_{}.csv".format(cls, "density"), index=False)
            #     print(" density at query point:",strategy,density, traditional_density)
            #     print("---------------")
        R[strategy]=recall_values
        df = pd.DataFrame(R)
        df.to_csv("output_{}_{}.csv".format(cls, "Recall"), index=False)
        P[strategy] = precision_values
        df = pd.DataFrame(P)
        df.to_csv("output_{}_{}.csv".format(cls, "Precision"), index=False)
        F1[strategy] = f1_values
        df = pd.DataFrame(F1)
        df.to_csv("output_{}_{}.csv".format(cls,"F1"), index=False)
        # print(json.dumps(recall_values))
        print("Estimated density's time:", usetime)

        # 绘制
        plot_results(k_values, recall_values, precision_values,title=strategy)

        # 使用PCA降维至2维
        # 将圆心与查询点一起进行PCA降维
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(points)  # 降维到2维

        # 计算距离矩阵
        # dist_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

        # # MDS降维到2D
        # mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        # data_2d = mds.fit_transform(dist_matrix)

        # 圆心和查询点在降维后的2D空间
        q_2d = data_2d[q_point]
        p_2d = data_2d

    # q_2d = query_point
    # p_2d = points

    # 绘制图形
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for r in radii:
        circle = plt.Circle((q_2d[0], q_2d[1]), r, fill=False, color='b', linestyle='--')
        ax.add_artist(circle)

    # 绘制查询点
    plt.scatter(p_2d[:, 0], p_2d[:, 1], color='r', label="Query Points")

    # 绘制圆心
    plt.scatter(q_2d[0], q_2d[1], color='g', label="Center (q)")

    # 设置坐标轴
    # plt.xlim(q_2d[0, 0] - max(radii) - 1, q_2d[0, 0] + max(radii) + 1)
    # plt.ylim(q_2d[0, 1] - max(radii) - 1, q_2d[0, 1] + max(radii) + 1)

    # 添加图例
    plt.legend()

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()