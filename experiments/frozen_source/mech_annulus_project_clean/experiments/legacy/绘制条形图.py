import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=15)

# 数据准备
#cifar10
data1 = {
    'Accurate Annulus': [60, 20, 20],
    'Approximate Annulus': [200,20, 20],
    'MCMC': [ 1000,500, 180]
}
#Amazon
data4 ={
    'Accurate Annulus': [20, 20, 20],
    'Approximate Annulus': [20,20, 20],
    'MCMC': [ 1100,20, 20]
}
#isolet
data3 = {
    'Accurate Annulus': [20, 20, 20],
    'Approximate Annulus': [20,20, 20],
    'MCMC': [ 100,40, 20]
}
#GIST-512
data2 = {
    'Accurate Annulus': [300, 40, 20],
    'Approximate Annulus': [600,60, 25],
    'MCMC': [ 2000,800, 280]
}
categories = ['0.001','0.003','0.005' ]

# 设置柱子宽度和位置
bar_width = 0.25  # 每组柱子的宽度
x = np.arange(len(categories))  # 类别位置 [0, 1, 2]

# 创建画布
plt.figure(figsize=(10, 6))
data=data4
plt.grid(linestyle='--')
cmap=  ['c','y','r']
# 绘制每组柱子
for i, (group_name, values) in enumerate(data.items()):
    # 计算每组柱子的x轴偏移量（居中显示）
    offset = bar_width * (i - (len(data) - 1) / 2)
    bars = plt.bar(x + offset, values, width=bar_width, label=group_name, color=cmap[i])

    # 在柱子上方添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, height,
    #              f'{height}', ha='center', va='bottom', fontsize=10)

# 自定义图表样式
plt.xticks(x, categories)  # 设置x轴标签
plt.xlabel("Relative Error")
plt.ylabel("Number of Samples")
# plt.title('Multi-Group Bar Chart Comparison', fontsize=14, pad=20)
plt.legend( loc='upper right')  # 图例放在右侧

# 添加网格和调整边距
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()  # 自动调整布局

plt.savefig("out_{}_bar.png",dpi=800)
# 显示图表
plt.show()
import  pandas as pd
plt.figure(figsize=(10, 8))
df = pd.DataFrame(data,index= ['0.001','0.003','0.005' ])
# 绘制堆叠柱状图
ax = df.plot.bar(stacked=True, width=0.3, figsize=(10, 7), color=cmap,rot=0)
plt.xlabel("Relative Error")
plt.ylabel("Number of Samples")
plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()  # 自动调整布局
plt.savefig("out_{}_bar1.png",dpi=800)
plt.show()