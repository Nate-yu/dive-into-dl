import torch
x = torch.arange(4.0,requires_grad=True)
# print(x) 
# x.requires_grad(True) # 等价于x=torch.arange(4.0,requires_grad=True)
# print(x.grad) # 默认值是None

# 计算y 
y = 2 * torch.dot(x,x) # x是一个长度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出。
# print(y)

# 通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
y.backward()
# print(x.grad)

# 快速验证梯度计算是否正确
# print(x.grad == 4*x)

# 现在计算x的另一个函数
x.grad.zero_()
y = x.sum()
y.backward()
# print(x.grad)

x.grad.zero_()
y = x*x
y.sum().backward() # 等价于y.backward(torch.ones(len(x)))
# y.backward(torch.ones(len(x))) # torch.ones(len(x))是一个长度为x的向量，其中每个元素都是1。这个向量告诉PyTorch我们要计算y关于x的梯度，并且每个元素的权重都是1。这个权重向量可以用来计算y关于x的加权和，其中每个元素的权重都是1。
# print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach() # 用于从计算图中分离出一个张量，梯度不会向后流经u到x
z = u * x
z.sum().backward()
# print(x.grad == u)

x.grad.zero_()
y.sum().backward()
# print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000: # 计算张量b的范数
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 计算梯度
a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
d.backward()
print(a.grad, a.grad == d / a)
