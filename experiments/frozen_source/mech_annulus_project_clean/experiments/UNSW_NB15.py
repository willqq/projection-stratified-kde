from sklearn.datasets import fetch_kddcup99
import numpy as np
import ssl
import  pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
ssl._create_default_https_context = ssl._create_unverified_context
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=15)
def gaussian_kde_np(X,x, bandwidth=0.5):
    """
    使用 NumPy 计算高斯核密度估计
    :param x: 待计算密度的点（标量或数组）
    :param data: 样本数据（1D 数组）
    :param bandwidth: 带宽（控制平滑程度）
    :return: 密度估计值
    """
    n, d = X.shape
    h = bandwidth

    # 计算平方欧氏距离 ||x - x_i||^2
    squared_dist = np.sum((x - X) ** 2, axis=1)  # 形状 (n,)

    # 计算高斯核密度
    kde_value = np.sum(np.exp(-squared_dist / (2 * h ** 2))) / (n * (h ** d) * (2 * np.pi) ** (d / 2))
    return kde_value
df = pd.read_csv('/Users/jingenyan/Downloads/UNSW_NB15_training-set 2.csv')
# df.info()
df=df[:10000]


list_drop = ['id','attack_cat']
df.drop(list_drop,axis=1,inplace=True)
df_numeric = df.select_dtypes(include=[np.number])
df_numeric=df_numeric.dropna(axis=0)
from sklearn.neighbors import KernelDensity
X=df_numeric.drop(['label'],axis=1)
Y=df_numeric['label']
X = (X - X.mean()) / X.std()
x=X.to_numpy()
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
# 示例数据 - 替换为你的实际数据
X = X # 100个样本，每个样本50维特征
labels = Y  # 100个样本的类别标签
# 执行T-SNE降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42,n_iter=2000)
X_embedded = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title('T-SNE Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
# kd = KernelDensity(kernel='gaussian', bandwidth=0.3)
# kd.fit(X[Y == 1])

x1=x[Y == 1]
x2=x[Y == 0]

n1=0
radius=20
radiuses = np.linspace(0, radius, 20)
huan_n = 0
from tqdm  import tqdm
Yn1= []
Ya1= []
# Yn =np.array([0])
# Ya=np.array([0])
Yn2= []
Ya2=[]

Yn=[]
Ya=[]
t1=time.time()
for index,i in enumerate(x1):
    distances = np.linalg.norm(x1 - i,axis=1)
    x = x1[distances<10]
    dens= gaussian_kde_np(x, i, bandwidth=0.3)
    Yn.append(dens)

for index, i in enumerate(x2):
    distances = np.linalg.norm(x1 - i, axis=1)
    dens = gaussian_kde_np(x1[distances<10], i, bandwidth=0.3)
    print(dens)
    Ya.append(dens)
result = np.vstack((np.array(x1), np.array(x2)))
print(np.array(Yn+ Ya) <5)
labels0 =np.array(np.array(Yn+ Ya) <5)
labels=[0]*len(Yn) + [1]*len(Ya)
X_embedded = tsne.fit_transform(result)



plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title('True Labels')

plt.show()
# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels0, cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title('Accurate Annulus')
plt.show()

print("t1",time.time()-t1)

t1=time.time()
distances1 = np.linalg.norm(x1, axis=1)
for index,i in enumerate(x1):
    distances = np.linalg.norm(i)
    x=x1[distances1<distances+10]
    dens= gaussian_kde_np(x, i, bandwidth=0.3)
    Yn1.append(dens)
n2=0
for index, i in enumerate(x2):
    distances = np.linalg.norm(x1 - i, axis=1)
    x = x1[distances1<distances+10]
    dens = gaussian_kde_np(x, i, bandwidth=0.3)
    Ya1.append(dens)
    # if dens <5:
    #     n2=n2+1

print("t2",time.time()-t1)

labels1 =np.array(np.array(Yn1+ Ya1) <5)
random_rows=x1[np.random.choice(x1.shape[0], size=1000, replace=False), :]
for index,i in enumerate(x1):
    distances = np.linalg.norm(x1 - i,axis=1)
    dens= gaussian_kde_np(random_rows, i, bandwidth=0.3)
    Yn2.append(dens)
for index, i in enumerate(x2):
    distances = np.linalg.norm(x1 - i, axis=1)
    dens = gaussian_kde_np(random_rows, i, bandwidth=0.3)
    Ya2.append(dens)
labels2 =np.array(np.array(Yn2+ Ya2) <5)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim(0, 5000)
# ax.plot_trisurf(X_embedded[:, 0], X_embedded[:, 1], Yn+ Ya, cmap=plt.get_cmap('jet'), linewidth=0.1)  # 三角表面图
# ax.scatter(xs,ys,zs)
ax.scatter(X_embedded[ 0:len(Yn),0], X_embedded[0:len(Yn),1], Yn,c='r',zorder=5,alpha=0.01)
ax.scatter(X_embedded[len(Yn):,0], X_embedded[len(Yn):,1], Ya,c='g',zorder=10,alpha=0.5)
plt.show()  # 渲染3D图显示
fig = plt.figure(dpi=100)
ax = Axes3D(fig)
ax.set_zlim(0, 5000)

ax.scatter(X_embedded[ 0:len(Yn),0], X_embedded[0:len(Yn),1], Yn1,c='r',alpha=0.01)
ax.scatter(X_embedded[len(Yn):,0], X_embedded[len(Yn):,1], Ya1,c='g',alpha=0.5)
plt.show()  # 渲染3D图显示
fig = plt.figure(dpi=100)
ax = Axes3D(fig)
ax.set_zlim(0, 5000)
ax.scatter(X_embedded[ 0:len(Yn),0], X_embedded[0:len(Yn),1], Yn2,c='r',alpha=0.01)
ax.scatter(X_embedded[len(Yn):,0], X_embedded[len(Yn):,1], Ya2,c='g',alpha=0.5)
plt.show()  # 渲染3D图显示

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels1, cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title('Approximate Annulus')
plt.show()
Ya=np.array(Ya)
Yn=np.array(Yn)
P=np.sum(Ya <5)/len(Ya)
R=np.sum(Ya <5)/(np.sum(Ya <5)+np.sum(Yn <5))
print("acc",P,R,2*P*R/(P+R))
Ya1=np.array(Ya1)
Yn1=np.array(Yn1)
P=np.sum(Ya1 <5)/len(Ya1)
R=np.sum(Ya1 <5)/(np.sum(Ya1 <5)+np.sum(Yn1 <5))
print("pro",P,R,2*P*R/(P+R))

# print( n1,n2,n1+n2,n2/len(x2),n2/(n2+len(x1)-n1),2*n2/len(x2)*n2/(n2+len(x1)-n1)/(n2/(n2+len(x1)-n1)+n2/len(x2)))

t4=time.time()

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels2, cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title('MCMC')
plt.show()

print("t4",time.time()-t4)
Ya2=np.array(Ya2)
Yn2=np.array(Yn2)
P=np.sum(Ya2 <0.02)/len(Ya2)
R=np.sum(Ya2 <0.02)/(np.sum(Ya2 <0.02)+np.sum(Yn2 <0.02))
print("mcmc",P,R,2*P*R/(P+R))



# kd = KernelDensity(kernel='gaussian', bandwidth=0.3)
# kd.fit(X[Y == 1])
# Yn = np.exp(kd.score_samples(X[Y == 1]))
# Ya = np.exp(kd.score_samples(X[Y != 1]))
# Yn2=np.array(Yn)
# Ya2=np.array(Ya)
for threshold in  np.linspace(0, 1000, 1000):
    P = np.sum(Ya < threshold) / len(Ya)
    R = np.sum(Ya < threshold) / (np.sum(Ya <threshold) + np.sum(Yn < threshold))
    print("threshold",threshold,P, R, 2 * P * R / (P + R))

bins=100
plt.figure(figsize=(10, 7))
plt.hist(Yn, bins=bins,  alpha=1,  color= (0.1, 0.5, 0.8),range=(0,2000),)
# 设置图表属性
plt.grid(linestyle='--')

plt.title('Normal Samples')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 7))
plt.hist(Ya, bins=bins, alpha=1, color= (0.1, 0.5, 0.8),range=(0,1000))
# 设置图表属性
plt.grid(linestyle='--')
plt.title('Abnormal Samples')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()

#近似环
plt.figure(figsize=(10, 7))
plt.hist(Yn1, bins=bins,  alpha=1, color= (0.1, 0.5, 0.8),range=(0,2000),)
# 设置图表属性
plt.grid(linestyle='--')
plt.title('Normal Samples')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 7))
plt.hist(Ya1, bins=bins, alpha=1, color= (0.1, 0.5, 0.8),range=(0,1000))
# 设置图表属性
plt.grid(linestyle='--')
plt.title('Abnormal Samples')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()

#MCMC
plt.figure(figsize=(10, 7))
plt.hist(Yn2, bins=bins,  alpha=1, color= (0.1, 0.5, 0.8),range=(0,2000),)
# 设置图表属性
plt.grid(linestyle='--')

plt.title('Normal Samples')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 7))
plt.hist(Ya2, bins=bins, alpha=1, color= (0.1, 0.5, 0.8),range=(0,1000))
# 设置图表属性
plt.grid(linestyle='--')
plt.title('Abnormal Samples')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()
from scipy.stats import pearsonr

# 计算Pearson相关系数和p值
Y1 = np.hstack((Ya1.flatten(), Yn1.flatten()))
Y=np.hstack((Ya.flatten(), Yn.flatten()))
mask = np.isfinite(Y1) & np.isfinite(Y)
# print([np.where(mask)])
corr, p_value = pearsonr(Y1[np.where(mask)], Y[np.where(mask)])

# 绘制散点图
plt.scatter(Y1[np.where(mask)], Y[np.where(mask)], alpha=0.6)
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-1000, 3000)
plt.ylim(-1000, 3000)
# 添加相关系数文本
plt.text(1, 9, f"Pearson r = {corr:.2f}\np-value = {p_value:.3f}",
         bbox={'facecolor':'white', 'alpha':0.8})
plt.title("Correlation Coefficient")
plt.show()
print('Mean normal: {:.5f} - Std: {:.5f}'.format(np.mean(Yn), np.std(Yn)))
print('Mean anomalies: {:.5f} - Std: {:.5f}'.format(np.mean(Ya), np.std(Ya)))
# Download latest version
# path = kagglehub.dataset_download("mishra5001/credit-card")
#
# print("Path to dataset files:", path)
# kddcup99 = fetch_kddcup99(subset='http', percent10=True, random_state=1000)
#
# X = kddcup99['data'].astype(np.float64)
# Y = kddcup99['target']
# print(X)
# print('Statuses: {}'.format(np.unique(Y)))
# print('Normal samples: {}'.format(X[Y == b'normal.'].shape[0]))
# print('Anomalies: {}'.format(X[Y != b'normal.'].shape[0]))
# means = np.mean(X, axis=0)
# stds = np.std(X, axis=0)
# IQRs = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
# N = float(X.shape[0])
#
# h0 = 0.9 * np.min([stds[0], IQRs[0] / 1.34]) * np.power(N, -0.2)
# h1 = 0.9 * np.min([stds[1], IQRs[1] / 1.34]) * np.power(N, -0.2)
# h2 = 0.9 * np.min([stds[2], IQRs[2] / 1.34]) * np.power(N, -0.2)
#
# print('h0 = {:.3f}, h1 = {:.3f}, h2 = {:.3f}'.format(h0, h1, h2))