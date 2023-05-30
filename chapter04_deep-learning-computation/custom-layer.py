# 1. 不带参数的层
import torch
import torch.nn.functional as F
from torch import nn

# 下面的CenteredLayer类要从其输入中减去均值
class CenteredLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    
# 向该层提供一些数据，验证是否能按预期工作
layer = CenteredLayer()
# print(layer(torch.FloatTensor([1,2,3,4,5])))

# 我们可以将层作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8,128), CenteredLayer())
Y = net(torch.rand(4,8))
# print(Y.mean())

# 2. 带参数的层
class MyLinear(nn.Module):
    # in_units和units，分别表示输入数和输出数
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

# 实例化MyLinear类并访问其模型参数
linear = MyLinear(5, 3)
# print(linear.weight)

# 使用自定义层直接执行前向传播计算
# print(linear(torch.rand(2,5)))

# 使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层
net = nn.Sequential(MyLinear(64,8), MyLinear(8,1))
print(net(torch.rand(2,64)))