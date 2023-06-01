import torch
from torch import nn
# 1. 填充
# 为了方便起见，我们定义了一个计算卷积层的函数。此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    X = X.reshape((1,1) + X.shape) # 这里的（1,1）表示批量大小和通道数都是1
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1)
X = torch.rand(size=(8,8))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

# 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度
conv2d = nn.Conv2d(1,1,kernel_size=(5,3),padding=(2,1))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

# 2. 步幅
# 将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半
conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2)
Y = comp_conv2d(conv2d, X)
print(Y.shape)

conv2d = nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
Y = comp_conv2d(conv2d,X)
print(Y.shape)