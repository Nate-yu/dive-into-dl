# 首先看一下具有单隐藏层的多层感知机
import torch
from torch import nn
net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2,4))
# print(net(X))

# 1. 参数访问
print(net[2].state_dict()) # 检查第二个全连接层的参数：这个全连接层包含两个参数，分别是该层的权重和偏置

# 从第二个全连接层（即第三个神经网络层）提取偏置， 提取后返回的是一个参数类实例，并进一步访问该参数的值。
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# 由于我们还没有调用反向传播，所以参数的梯度处于初始状态
print(net[2].weight.grad == None) # True

# 比较访问第一个全连接层的参数和访问所有层
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 另一种访问网络参数的方式
print(net.state_dict()['2.bias'].data)

# 从嵌套块收集参数: 定义一个生成块的函数（可以说是“块工厂”），然后将这些块组合到更大的块中
def block1():
    return nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 嵌套block1
        net.add_module(f'block{i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4,1))
print(rgnet(X))

# 设计了网络后，我们看看它是如何工作的
print(rgnet)

# 访问第一个主要的块中、第二个子块的第一层的偏置项
print(rgnet[0][1][0].bias.data)

# 2. 参数初始化
# 首先调用内置的初始化器，将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])

# 还可以将所有参数初始化为给定的常数，比如初始化为1
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

# 使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data[0])

# 实现一个my_init函数来应用到net
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)
print(net[0].weight[:2])

# 我们始终可以直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

# 3. 参数绑定: 可以定义一个稠密层，然后使用它的参数来设置另一个层的参数
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8,1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0,0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])