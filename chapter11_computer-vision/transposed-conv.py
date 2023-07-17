import torch
from torch import nn
from d2l import torch as d2l

# 基本操作
# 对输入矩阵X和卷积核矩阵K实现基本的转置卷积运算
def trans_conv(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i+h, j: j+w] += X[i,j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(trans_conv(X,K))

# 使用高级API获得相同的结果
X, K = X.reshape(1,1,2,2), K.reshape(1,1,2,2)
tconv = nn.ConvTranspose2d(1,1,kernel_size=2,bias=False)
tconv.weight.data = K
# print(tconv(X))

# 填充、步幅和多通道
# 当将高和宽两侧的填充数指定为1时，转置卷积的输出中将删除第一和最后的行与列
tconv = nn.ConvTranspose2d(1,1,kernel_size=2,padding=1,bias=False)
tconv.weight.data = K
# print(tconv(X))

# 将步幅从1更改为2会增加中间张量的高和权重
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
# print(tconv(X))

X = torch.rand(size=(1,10,16,16))
conv = nn.Conv2d(10,20,kernel_size=5,padding=2,stride=3)
tconv = nn.ConvTranspose2d(20,10,kernel_size=5,padding=2,stride=3)
# print(tconv(conv(X)).shape == X.shape)

# 与矩阵变换的联系
X = torch.arange(9.0).reshape(3,3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X,K)
# print(Y)

# 将卷积核K重写为包含大量0的稀疏权重矩阵W
def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4,9))
    k[:2],k[3:5] = K[0,:], K[1,:]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
# print(W)

# print(Y == torch.matmul(W, X.reshape(-1)).reshape(2,2))

# 可以使用矩阵乘法来实现转置卷积
Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3,3))