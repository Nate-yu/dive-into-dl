import numpy as np
from matplotlib import pyplot as plt
from d2l import torch as d2l
# from calculus import use_svg_display, set_figsize, set_axes, plot

# 1. 绘制函数y = f(x) = x^3 - 1/x 和其在x = 1处切线的图像

# 方式一
# def f(x):
#     return x**3 - 1.0/x

# x = np.arange(0,3,0.1)
# plot(x,[f(x),4*x-4],'x','f(x)',legend=['f(x)','Tangent line(x=1)'])
# d2l.plt.show()

# 方式二
def get_fun(x):
    return x**3 - 1/x

def get_tangent(f, x, point):
    h = 1e-4
    grad = (f(point+h) - f(point)) / h
    return grad*(x-point) + f(point)

x = np.arange(0.1,3.0,0.01)
y = get_fun(x)
y_tangent = get_tangent(get_fun, x, 1)
plt.plot(x,y)
plt.plot(x,y_tangent)
plt.show()