import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# data = pd.read_excel("数据.xlsx")
# 全部的线段风格

# 设置字体
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=15)
styles = ['c:s','y:8','r:^','g:D','r:v','m:X','b:p',':>'] # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'
# 获取全部的图例
x=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# x={"Amazon":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],"Amazon1":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],"Amazon2":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],"Amazon3":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
# y={"Amazon":[0.37777777777777777, 0.5644444444444444, 0.6622222222222223, 0.8733333333333333, 0.92, 0.9555555555555556, 0.9688888888888889, 0.9733333333333334, 0.9955555555555555, 0.9955555555555555],"Amazon1":[0.06888888888888889, 0.19555555555555557, 0.30666666666666664, 0.5577777777777778, 0.7155555555555555, 0.8177777777777778, 0.8577777777777778, 0.9111111111111111, 0.9533333333333334, 0.9577777777777777],"Amazon2":[0.0022222222222222222, 0.0022222222222222222, 0.06666666666666667, 0.18666666666666668, 0.4311111111111111, 0.7155555555555555, 0.7711111111111111, 0.9222222222222223, 0.9844444444444445, 0.9888888888888889],"Amazon3":[0.6, 0.7711111111111111, 0.86, 0.9088888888888889, 0.9333333333333333, 0.9555555555555556, 0.9955555555555555, 0.9955555555555555, 0.9977777777777778, 0.9977777777777778]}
#
# x={"cifar10":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],"cifar101":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],"cifar102":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],"cifar103":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
y={"cifar10":[0.37777777777777777, 0.5644444444444444, 0.6622222222222223, 0.8733333333333333, 0.92, 0.9555555555555556, 0.9688888888888889, 0.9733333333333334, 0.9955555555555555, 0.9955555555555555],"cifar101":[0.06888888888888889, 0.19555555555555557, 0.30666666666666664, 0.5577777777777778, 0.7155555555555555, 0.8177777777777778, 0.8577777777777778, 0.9111111111111111, 0.9533333333333334, 0.9577777777777777],"cifar102":[5.555555555555556e-05, 0.0009444444444444445, 0.034444444444444444, 0.32572222222222225, 0.4225555555555556, 0.7912777777777777, 0.9287222222222222, 0.9806666666666667, 0.9800555555555556, 0.992],"cifar103":[0.4666111111111111, 0.7809444444444444, 0.874, 0.9835555555555555, 0.9865, 0.9910555555555556, 0.9931111111111111, 0.9937777777777778, 0.9955555555555555, 0.9993888888888889]}
for clor,name in enumerate(["Recall","Precision","F1"]):
    data = pd.read_csv('output_cifar10_{}.csv'.format(name))  # 替换为你的文件路径
# columns = ["HyperplaneLSH+Hamming","SimHash","HyperplaneLSH only","AngularLSH"]
# 正式的进行画图
    plt.figure(figsize=(10, 7))
    plt.xlim([0, 22])
    # plt.ylim([0, 1])
    plt.xlabel("Number of Hash Table")
    plt.ylabel(name)
    column_names = data.columns.tolist()
    for index,i in enumerate(column_names):

        plt.plot(x,data[i], styles[index], linewidth =2.0,markersize=8,label=i)
        # 设置图片的x,y轴的限制，和对应的标签


        # 设置图片的方格线和图例
    plt.grid(linestyle = '--')
    if name == "Precision":
        plt.legend(loc='upper right',framealpha=0.7)
    else:
        plt.legend(loc='lower right',framealpha=0.7)
    plt.tight_layout()
    plt.savefig("out_{}.png".format(name),dpi=800)
    plt.show()
data = pd.read_csv('output_Amazon_density.csv')  # 替换为你的文件路径
# columns = ["HyperplaneLSH+Hamming","SimHash","HyperplaneLSH only","AngularLSH"]
# 正式的进行画图
plt.figure(figsize=(10, 7))
# plt.xlim([0, 22])
# plt.ylim([0, 1])
plt.xlabel("Number of Samples")
plt.ylabel("Relative Error")
column_names = data.columns.tolist()
for index,i in enumerate(column_names[1:]):#[Accurate Annulus,Approximate Annulus,MCMC]
    name=["Accurate Annulus","Approximate Annulus","MCMC"]
    ys=[]
    yys=[]
    nums=[]
    for k,y in enumerate(data[i]):
        ys.append(y)
        if data[column_names[0]][k]%100==0:
            nums.append(data[column_names[0]][k])
            yys.append(np.mean(ys)/10)
    print(i,yys)



    plt.plot(nums,yys, styles[index], linewidth =2.0,markersize=8,label=name[index])
    # 设置图片的x,y轴的限制，和对应的标签


    # 设置图片的方格线和图例
plt.grid(linestyle = '--')

plt.legend(loc='upper right',framealpha=0.7)

# else:
#     plt.legend(loc='lower right',framealpha=0.7)
plt.tight_layout()
plt.savefig("out_{}.png".format(name),dpi=800)
plt.show()

# 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
