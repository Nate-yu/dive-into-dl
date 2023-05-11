import torch
x = torch.arange(12)
print(x,x.shape,x.numel()) # numel()表示一个标量，表示的是张量中元素个数

X = x.reshape(3,4)
print(X)

print(torch.zeros(2,3,4)) # 输出2个3行4列全为0的张量
print(torch.ones(1,3,4)) # 输出1个3行4列全为1的张量

print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])) # 为张量赋初值

print(torch.randn(3,4)) # 创建一个形状为(3,4)的张量，每个元素都从均值为0，标准差为1的高斯分布中随机采样

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x+y,'\n',x-y,'\n',x*y,'\n',x/y,'\n',x**y)
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0)) # 按行顺序组合
print(torch.cat((X, Y), dim=1)) # 按列顺序组合

print(X == Y, '\n', X > Y, '\n', X < Y)

X[0:2, :] = 12
print(X)

# 对于可变对象而言，x=x+y产生的新的x保存在新的内存中，而x+=y是原地操作。
before = id(Y) # 张量Y在Python中的唯一标识
print(before)
# Y = Y + X # False
Y += X # True
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z): ',id(Z))
Z[:] = X + Y
print('id(Z): ',id(Z))

A = X.numpy()
B = torch.tensor(A)
print(type(A),type(B))

a = torch.tensor([3.5])
print(a,a.item(),float(a),int(a))