import torch
from torch import nn
from d2l import torch as d2l
# 1. 互相关运算
# 计算二维互相关运算，该函数接受输入张量X和卷积核张量K，并返回输出张量Y
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i : i+h, j : j+w] * K).sum()
    return Y
# 验证上述二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(corr2d(X, K))

# 2. 卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
# 3. 图像中目标的边缘检测
X = torch.ones((6,8))
X[:, 2:6] = 0
# print(X)
K = torch.tensor([[1.0,-1.0]]) # 构造一个高度为1宽度为2的卷积核K，当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零。
Y = corr2d(X, K) # 对参数X（输入）和K（卷积核）执行互相关运算。输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0
# print(Y)

# 这个卷积核K只可以检测垂直边缘，无法检测水平边缘
# print(corr2d(X.t(), K))

# 4. 学习卷积核
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1,2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch{i+1}, loss {l.sum():.3f}')

# 现在我们来看看我们所学的卷积核的权重张量
print(conv2d.weight.data.reshape((1,2))) # 现在我们来看看我们所学的卷积核的权重张量