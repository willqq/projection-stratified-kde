import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv('data/气泡图.csv')
print(data)
# 绘制气泡图（用hue参数可添加颜色维度）
sns.scatterplot(
    data=data,
    x='技术竞争力',
    y='业务竞争力',
    size='bubble_size',
    sizes=(50, 500),  # 气泡大小范围
    hue='技术元素',    # 按类别区分颜色
    alpha=0.7
)

# 调整图例位置
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

# 显示图表
plt.title('Seaborn气泡图')
plt.tight_layout()
plt.show()