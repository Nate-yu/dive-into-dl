import torch
from torch import nn
from d2l import torch as d2l
# 实现池化层的前向传播
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i : i+p_h, j : j+p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i : i+p_h, j : j+p_w].mean()
    return Y

""" # 构建输入张量X，验证二维最大池化层的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2,2)))
# 验证平均池化层的输出
print(pool2d(X, (2,2), mode='avg')) """

# 下面用深度学习框架中内置的二维最大池化层，来演示池化层中填充和步幅的使用
# 首先构造了一个输入张量X，它有四个维度，其中样本数和通道数都是1
X = torch.arange(16, dtype=torch.float32).reshape((1,1,4,4))
print(X)
# 默认情况下，深度学习框架中的步幅与池化窗口的大小相同
pool2d = nn.MaxPool2d(3)
print(pool2d(X))

# 填充和步幅可以手动设定
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

# 可以设定一个任意大小的矩形池化窗口，并分别设定填充和步幅的高度和宽度
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))

# 在处理多通道输入数据时，池化层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总
# 下面在通道维度上连结张量X和X + 1，以构建具有2个通道的输入
X = torch.cat((X, X+1),1) # 第二个参数1代表按顺序输出，如果为0则按tensor的大小输出
print(X)

# 如下所示，池化后输出通道的数量仍然是2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))