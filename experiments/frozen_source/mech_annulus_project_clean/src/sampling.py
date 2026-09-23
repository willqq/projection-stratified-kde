import  numpy as np
#核密度估计函数
def kernel_density_estimation(data: np.ndarray, bandwidth: float) :
    """
    核密度估计函数
    :param data: 输入数据
    :param bandwidth: 带宽参数
    :return: x轴和对应的密度值
    """
    x_min = np.min(data)
    x_max = np.max(data)
    x = np.linspace(x_min, x_max, 1000)
    density = np.zeros_like(x)
    for i in range(len(x)):
        density[i] = np.sum(np.exp(-0.5 * ((x[i] - data) / bandwidth) ** 2)) / (bandwidth * np.sqrt(2 * np.pi))
    return x, density