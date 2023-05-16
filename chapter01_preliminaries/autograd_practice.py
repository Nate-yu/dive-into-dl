import torch
import numpy as np
import matplotlib.pyplot as plt
# 4. 重新设计一个求控制流梯度的例子，运行并分析结果。
def f(a):
    if a.norm() > 10:
        b = a
    else:
        b = 2 * a
    return b.sum()

a = torch.arange(6.0, requires_grad=True)
d = f(a)
d.backward()
print(a.grad)

# 5. 使f(x) = sinx，绘制f(x)和其导数的图像，后者不能使用f'(x) = cosx
# 数据初始化
x = np.arange(-5, 5, 0.02)
f = np.sin(x)
df = []

# 利用反向传播，求出x的梯度
for i in x:
    v = torch.tensor(i, requires_grad=True)
    y = torch.sin(v)
    y.backward()
    df.append(v.grad)

# 绘图
fig, ax = plt.subplots()
ax.plot(x, f, 'k', label = 'f(x)')
ax.plot(x, df, 'k*', label = 'df(x)')
# 美化legend
legend = ax.legend(loc = 'upper left', shadow = True, fontsize = 'x-large')
legend.get_frame().set_facecolor('C0') 

plt.show()