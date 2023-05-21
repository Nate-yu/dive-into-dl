import torch
from torch import nn
from d2l import torch as d2l

# 1. 模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:  
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10 	# 参数设置为 256 ，0.1， 10 次迭代
loss = nn.CrossEntropyLoss(reduction='none') # 定义损失函数为交叉熵函数
trainer = torch.optim.SGD(net.parameters(), lr=lr) # 定义优化器为随机梯度下降法
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 加载数据集和测试
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) # 开始计算准确率