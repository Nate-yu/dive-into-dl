import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

""" h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}') # 下面是等价语句
    # print('h={:.5f}, numerical limit={:.5f}'.format(h,numerical_lim(f, 1, h)))
    h *= 0.1 """

# 使用svg格式显示绘图
def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

# 设置matplotlib的图表大小
def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

# 设置由matplotlib生成图表的轴的属性
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid() # 用于在图形中添加网格线

# 定义一个plot函数来简洁地绘制多条曲线
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca() # gca()用于获取当前的 matplotlib 图形对象中的坐标轴对象

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        # hasattr()来判断X是否有属性"ndim", isinstance()函数来判断X是否是一个列表, 如果X不是一维数组或一维列表，那么函数会返回False
        return(hasattr(X,"ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0],"__len__"))
    
    if has_one_axis(X):
        X = [X] # 将X转换成一个列表
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x,y,fmt in zip(X, Y, fmts): # zip函数将X和Y中的每个元素一一对应
        if len(X):
            axes.plot(x,y,fmt)
        else:
            axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# 绘制函数u=f(x)及其在x=1处的切线y=2x-3
x = np.arange(0,3,0.1)
plot(x,[f(x),2*x-3],'x','f(x)',legend=['f(x)','Tangent line(x=1)'])
# d2l.plt.show()