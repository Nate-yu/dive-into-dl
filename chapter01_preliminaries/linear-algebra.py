import torch

# 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x**y)

# 向量
x = torch.arange(4)
print(x,x[3],len(x),x.shape)

# 矩阵
A = torch.arange(20).reshape(5,4)
print(A)
print(A.T)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

# 张量
X = torch.arange(24).reshape(2,3,4)
print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5,4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
print(A,'\n',A+B,'\n',A*B)

# 降维
x = torch.arange(4, dtype=torch.float32)
print(x,',',x.sum())

print(A.shape,',',A.sum()) # torch.Size([5, 4]) , tensor(190.)

A_sum_axis0 = A.sum(axis=0)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis0, A_sum_axis0.shape)
print(A_sum_axis1, A_sum_axis1.shape)
print(A.sum(axis=[0,1]))

print(A.mean(axis=0))
print(A.sum(axis=0)/A.shape[0])

# 非降维求和
sum_A = A.sum(axis=1, keepdim=True)
print(sum_A)
print(A / sum_A)

# 累计总和
print(A.cumsum(axis=0))

# 点积
y = torch.ones(4,dtype=torch.float32)
print(x,y,torch.dot(x,y))

# 向量积
print(A.shape,x.shape,torch.mv(A,x)) # 使用torch.mv()函数时，A的列维数必须与x的维数相同

# 矩阵乘法
B = torch.ones(4,3)
print(A,B)
print(torch.mm(A,B))

# 范数
u = torch.tensor([3.0,-4.0])
print(torch.norm(u)) # 5
print(torch.abs(u).sum()) # 7
print(torch.norm(torch.ones((4,9)))) # 6