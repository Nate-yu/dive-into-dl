import torch
from d2l import torch as d2l 

# 1. ReLU函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 3))
# d2l.plt.show()

y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize= (5, 3))
# d2l.plt.show()

# 2. sigmoid函数
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x','sigmoid(x)',  figsize=(5, 3))
# d2l.plt.show()

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigma(x)',  figsize=(5, 3))
# d2l.plt.show()

# 3. tanh函数 (正弦和余弦) （对于计算梯度可以使用relu函数）
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)',  figsize=(5, 3))
# d2l.plt.show()

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh(x)',  figsize=(5, 3))
# d2l.plt.show()

