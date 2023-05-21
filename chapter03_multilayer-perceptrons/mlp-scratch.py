import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1. 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256 # 输入、输出、隐藏化层
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01) # W1 = W1.
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) # b1 = b1
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01) # W2 = W2.
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True)) # b2 = b2.
params = [W1, b1, W2, b2] # 将params列表传递给优化器

# 2. 激活函数
def relu(X):
    a = torch.zeros_like(X) # 创建一个与x相同维度的零向量，并将其作为输入。
    return torch.max(X, a) 	# 大于零的值保持不变，小于零的值转换为零。

# 3. 模型
def net(X):
    X = X.reshape(-1, num_inputs) 
    H = relu(X @ W1 + b1)
    return H @ W2 + b2

# 4. 损失函数
loss = 	nn.CrossEntropyLoss(reduction='none') # 选择正确的交叉熵损失函数

# 5. 训练
num_epochs, lr = 10, 0.1 # 学习率为0.1的10次迭代次数
updater = torch.optim.SGD(params, lr=lr) # 使用参数更新器SGD
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.predict_ch3(net, test_iter)
d2l.plt.show()