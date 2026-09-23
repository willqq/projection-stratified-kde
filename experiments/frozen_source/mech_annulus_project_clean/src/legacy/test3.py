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
torch.manual_seed(seed)  # 设置CPU的随机种子
np.random.seed(seed)     # 设置NumPy的随机种子
random.seed(seed)        # 设置Python内置的随机种子

def min_max_scaler(original_data):
    scaler = MinMaxScaler()
    scaler.fit(original_data)
    normalized_data = pd.DataFrame(original_data).fillna(0).values
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
def build_hash_tables(points, filtered_candidates, num_tables, num_planes, dimension):
    hash_tables = []
    hyperplane_sets = []

    for _ in range(num_tables):
        hyperplanes = generate_random_hyperplanes(dimension, num_planes)  # 每个表使用不同的随机超平面
        hyperplane_sets.append(hyperplanes)  # 保存随机超平面
        hash_table = defaultdict(list)

        for point_id in filtered_candidates:
            point = points[point_id]
            hash_value = hash_point(point, hyperplanes)  # 计算点的哈希值
            hash_table[hash_value].append(point_id)  # 点的索引存入对应的哈希桶

        hash_tables.append(hash_table)

    return hash_tables, hyperplane_sets


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

def query_candidate_points(query_hash_value, hash_tables, distance_tables, max_hamming_distance=1):
    """根据哈希值查询候选点"""
    candidate_points = set()

    # 从每个哈希表中查询哈希值相同或汉明距离在阈值以内的点
    for table, distance_table in zip(hash_tables, distance_tables):
        if query_hash_value in table:
            candidate_points.update(table[query_hash_value])  # 查询哈希值相同的点

        # 通过汉明距离查询可能的点
        for hamming_distance in range(1, max_hamming_distance + 1):
            if query_hash_value in distance_table:
                for candidate_hash in distance_table[query_hash_value].get(hamming_distance, []):
                    candidate_points.update(table[candidate_hash])  # 查询与给定哈希值汉明距离为 i 的点

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

            # 如果角度小于最大角度，保留该候选点
            if angle <= max_angle_value:
                filtered_candidates.append(candidate_index)

    return filtered_candidates


def filter_candidates_by_distance(candidates, points, query_point, radius):
    """通过欧式距离过滤候选点"""
    filtered_candidates = []
    for candidate_index in candidates:
        candidate_point = points[candidate_index]  # 假设 points 是预处理过的所有数据点
        distance = np.linalg.norm(candidate_point - query_point)  # 计算欧式距离
        if distance <= radius:
            filtered_candidates.append(candidate_index)
    return filtered_candidates


def query_with_max_angle(points, query_point, radius, sphere_table, k, num_planes, dim, max_hamming_distance=1):
    """执行查询，考虑最大角度限制"""

    candidate_points = set()

    max_angle_value = max_angle(query_point, radius)

    # 步骤 4: 过滤候选点（根据角度）
    filtered_candidates = filter_candidates_by_angle(points, query_point, radius, sphere_table, max_angle_value)

    filtered_candidates = filter_candidates_by_distance(filtered_candidates, points, query_point, radius)

    hash_tables, hyperplane_sets = build_hash_tables(points, filtered_candidates, k, num_planes, dim)
    distance_tables = build_distance_tables(hash_tables)

    for i, hyperplane_set in enumerate(hyperplane_sets):
        query_hash_value = hash_point(query_point, hyperplane_set)  # 计算查询点在第 i 个哈希表中的哈希值
        candidate_points.update(
            query_candidate_points(query_hash_value, hash_tables, distance_tables, max_hamming_distance))


    return candidate_points



# ------------------------ Step 3: Approximate Annulus Partition ------------------------

def approximate_ring_query_multiple_radii(points, query_point, radii, sphere_table, k, num_planes, dim, max_hamming_distance=1):
    """
    执行多个半径的近似环查询。
    radii: 包含多个半径 [r1, r2, r3, ..., rn]
    """
    # 初始化结果集合
    candidate_sets = []

    # 步骤 1: 查询每个半径的近邻集合
    for radius in radii:
        candidate_points = query_with_max_angle(points, query_point, radius, sphere_table, k, num_planes, dim, max_hamming_distance)
        candidate_sets.append(set(candidate_points))  # 将结果存入集合

    # 步骤 2: 计算每对相邻半径的差集
    ring_queries = []
    for i in range(len(radii) - 1):
        r1_set = candidate_sets[i]
        r2_set = candidate_sets[i + 1]
        ring_queries.append(list(r2_set - r1_set))  # 计算 C(r2) - C(r1)

    return ring_queries  # 返回所有的近似环查询结果


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

def traditional_density_estimator(query_point, points, bandwidth=1.0):
    """传统的核密度估计，计算所有点的核密度"""
    density = 0.0
    n = len(points)
    for p in points:
        density += gaussian_kernel(query_point, p, bandwidth)
    return density / n  # 核密度的均值

# ------------------------ 实验指标 ------------------------

def get_true_neighbors(query_point, points, radius):
    """
    计算与查询点距离小于给定半径的真实邻居点数量
    """
    distances = np.linalg.norm(points - query_point, axis=1)
    true_neighbors = np.sum(distances <= radius)
    return true_neighbors

def compute_recall(true_neighbors, retrieved_neighbors):
    """
    计算召回率
    """
    return retrieved_neighbors / true_neighbors if true_neighbors > 0 else 0

def compute_precision(retrieved_neighbors, total_retrieved_neighbors):
    """
    计算准确率
    """
    return retrieved_neighbors / total_retrieved_neighbors if total_retrieved_neighbors > 0 else 0


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

def plot_results(x, recall, precision, candidate_accuracy):
    plt.figure(figsize=(10, 6))
    plt.plot(x, recall, marker='o', label="Recall", color='blue')
    plt.plot(x, precision, marker='s', label="Precision", color='green')
    plt.plot(x, candidate_accuracy, marker='^', label="Candidate Accuracy", color='red')
    plt.xlabel("k (Hash Code Length)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.show()

# ------------------------ Example Usage ------------------------

# Generate random data (1000 samples, 10 dimensions)
delta = 1
num_tables = 2
num_planes = 25
# points = np.random.rand(1000, 10)  # 1000个128维的点

# points = get_cifar_10_data()
# points = get_isolet_data()
points = get_Amazon_data()
# points = get_cifar10_Gist512_data()

print(points.shape)
dim = points.shape[1]
# query_point = np.random.rand(dim)
q_point = np.random.choice(points.shape[0])
query_point = points[q_point]
r1, r2, r3 = calculate_ring_parameters(query_point, points, 10, 30, 50)
radii = [r1, r2]
radius = r3
print("r1:", r1, "r2:", r2, "r3:", r3)
print("---------------")

points, query_point = translate_points_to_radius(query_point, radii[-1], points)

#传统核密度估计

starttime = datetime.datetime.now()
traditional_density = traditional_density_estimator(query_point, points, bandwidth=1.0)
print("Traditional density at query point:", traditional_density)
endtime = datetime.datetime.now()
print("Traditional density's time:", (endtime - starttime).seconds)
print("---------------")

usetime = 0

# 第一步：构建数据结构
sphere_table = build_sphere_table(points, delta)

# 假设有不同的k值
k_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
recall_values = []
precision_values = []
candidate_accuracy_values = []

for k in k_values:

    starttime_k = datetime.datetime.now()

    ring_queries = approximate_ring_query_multiple_radii(points, query_point, radii, sphere_table, k, num_planes, dim, max_hamming_distance=1)

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

    recall = compute_recall(true_neighbors, retrieved_neighbors)
    precision = compute_precision(retrieved_neighbors, total_retrieved_neighbors)
    candidate_accuracy = get_candidate_accuracy(query_point, candidate_set, radius)

    recall_values.append(recall)
    precision_values.append(precision)
    candidate_accuracy_values.append(candidate_accuracy)

print("Estimated density's time:", usetime)

# 绘制
plot_results(k_values, recall_values, precision_values, candidate_accuracy_values)

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