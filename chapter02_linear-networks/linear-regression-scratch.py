import random
import torch
from d2l import torch as d2l

# 生成数据集
def synthetic_data(w, b, num_examples):
    # 生成y=Xw+b+c噪声
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b # matmul()计算X和w之间的矩阵乘积
    y += torch.normal(0, 0.01, y.shape) # 生成一个形状为(numexamples, len(w))的张量X，其中每个元素都是从均值为0，标准差为1的正态分布中随机采样得到的
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）
# print('features: ', features[0], '\nlabel: ', labels[0])

# 生成第二个特征features[:, 1]和labels的散点图
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 这些样本是随机读取的，没有特定的顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])  # 取min是为了防止在最后一组数据时数据跨度小于batch_size
        yield features[batch_indices], labels[batch_indices] # 返回当前batch的features和labels

# 读取第一个小批量数据样本并打印
batch_size = 10
""" for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break """

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad(): # 创建了一个上下文环境，以确保在执行代码块时，不会计算梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')
