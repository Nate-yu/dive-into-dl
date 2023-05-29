# 回顾多层感知机: 下面的代码生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层， 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层
import torch
from torch import nn
from torch.nn import functional as F
net = nn.Sequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
# print(net(X))

# 自定义块：包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 以X作为输入， 计算带有激活函数的隐藏表示，并输出其未规范化的输出值
        return self.out(F.relu(self.hidden(X)))
    
net = MLP()
# print(net(X))

# 顺序块：下面的MySequential类提供了与默认Sequential类相同的功能
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中，_module的类型是OrderedDict
            self._modules[str(idx)] = module
    
    # 当MySequential的前向传播函数被调用时， 每个添加的块都按照它们被添加的顺序执行。
    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(X))

# 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
# 实现了一个隐藏层， 其权重（self.rand_weight）在实例化时被随机初始化，之后为常量
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20,20), requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight)+1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
net = FixedHiddenMLP()
# print(net(X))

# 我们以一些想到的方法嵌套块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))
    
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))