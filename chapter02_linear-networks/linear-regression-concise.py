# 生成数据集
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l 

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    # 构造一个PyTorch数据迭代
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))

# 定义模型
from torch import nn
net = nn.Sequential(nn.Linear(2,1)) # 将两个参数传递到nn.Linear中，第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # 指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样
net[0].bias.data.fill_(0) # 偏置参数将初始化为零

# 定义损失函数
loss = nn.MSELoss() # 返回所有样本损失的平均值

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    # print(f'epoch {epoch+1}, loss {l:f}')

w = net[0].weight.data
print('w的估计差：',true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计差：',true_b - b)