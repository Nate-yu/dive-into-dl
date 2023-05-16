# 矢量化加速
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

# 实例化两个全为1的10000维向量
n = 10000
a = torch.ones([n])
b = torch.ones([n])

# 定义一个计时器，记录多次运行时间
class Timer:
    
    def __init__(self):
        self.times = []
        self.start()

    # 启动计时器
    def start(self):
        self.tik = time.time()

    # 停止计时器并将时间记录在列表中
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    # 返回平均时间
    def avg(self):
        return sum(self.times) / len(self.times)
    
    # 返回时间总和
    def sum(self):
        return sum(self.times)
    
    # 返回累计时间
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
# 对工作负载进行基准测试
# 1. 使用for循环，每次执行一位的加法
c = torch.zeros(n)
timer = Timer()
""" for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec') """

# 2. 使用重载的+运算符来计算按元素的和
timer.start()
d = a+b
# print(f'{timer.stop():.5f} sec')

# 定义一个Python函数来计算正态分布
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)

# 可视化正态分布
x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)] # 均值和标准差对
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(5, 3), legend=[f'mean{mu}, std{sigma}' for mu, sigma in params])
d2l.plt.show()