import numpy as np
from typing import List
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
def min_max_scaler(original_data):
    scaler = MinMaxScaler()
    scaler.fit(original_data)
    normalized_data = pd.DataFrame(scaler.transform(original_data)).fillna(0).values
    return normalized_data
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
class HyperplaneLSH:
    def __init__(self, dim: int, k: int = 10, L: int = 5):
        """
        :param dim: 数据维度
        :param k: 每个哈希函数的超平面数量
        :param L: 哈希表数量(用于提高召回率)
        """
        self.dim = dim
        self.k = k
        self.L = L
        self.hash_tables = [{} for _ in range(L)]
        self.hyperplanes = [np.random.randn(k, dim) for _ in range(L)]

    def _hash(self, vec: np.ndarray, hyperplanes: np.ndarray) -> str:
        """计算单个哈希表的哈希键"""
        projections = np.dot(hyperplanes, vec)
        bits = (projections > 0).astype(int)
        return ''.join(bits.astype(str))

    def insert(self, vec: np.ndarray, id: str):
        """插入向量到所有哈希表"""
        for i in range(self.L):
            key = self._hash(vec, self.hyperplanes[i])
            if key not in self.hash_tables[i]:
                self.hash_tables[i][key] = []
            self.hash_tables[i][key].append(id)

    def query(self, vec: np.ndarray, top_k: int = 5) -> List[str]:
        """查询近似最近邻"""
        candidates = set()
        for i in range(self.L):
            key = self._hash(vec, self.hyperplanes[i])
            if key in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][key])
        return list(candidates)[:top_k]
points = get_cifar_10_data()
# points = get_isolet_data()
# points = get_Amazon_data()
# points = get_cifar10_Gist512_data()
mean = np.mean(points, axis=0)

# 中心化处理
points = points - mean
q_point = np.random.choice(points.shape[0])
query_point = points[q_point]
Hyperplane=HyperplaneLSH(618)
for i in range(points.shape[0]):
    point=points[i]
    Hyperplane.insert(points[i], str(i))
