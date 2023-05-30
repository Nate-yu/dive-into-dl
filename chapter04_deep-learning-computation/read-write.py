# 1. 加载和保存张量: 可以直接调用load和save函数分别读写它们。 这两个函数都要求我们提供一个名称，save要求将要保存的变量作为输入
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, './file/x-file')
# 将存储在文件中的数据读回内存
x2 = torch.load('./file/x-file')
# print(x2)

# 存储一个张量列表，然后把它们读回内存
y = torch.zeros(4)
torch.save([x, y], './file/x-files')
x2, y2 = torch.load('./file/x-files')
# print(x2, y2)

# 甚至可以写入或读取从字符串映射到张量的字典
mydict = {'x':x, 'y':y}
torch.save(mydict,'./file/mydict')
mydict2 =  torch.load('./file/mydict')
# print(mydict2)

# 2. 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
# 将模型的参数存储在一个叫做“mlp.params”的文件中
torch.save(net.state_dict(), './file/mlp.params')
# 为了恢复模型，我们实例化了原始多层感知机模型的一个备份。这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。
clone = MLP()
clone.load_state_dict(torch.load('./file/mlp.params'))
# print(clone.eval())

# 由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同
Y_clone = clone(X)
print(Y_clone == Y)
