""" # 数据准备
import os

os.makedirs(os.path.join('data'), exist_ok=True)
data_file = os.path.join('data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')  
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd
data = pd.read_csv(data_file)
inputs, outputs = data.iloc[:,0:2], data.iloc[:,2]

# 1. 删除缺失值最多的列

# 找出每列缺失值的个数
nan_num = inputs.isnull().sum(axis=0)
# 找出缺失值最多的列的索引
nan_max_id = nan_num.idxmax()
# 删除缺失值最多的那一列
inputs = inputs.drop([nan_max_id],axis=1)
print(inputs)

# 2. 将预处理后的数据集转换为张量格式
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y) """

import torch
# 1. 证明一个矩阵A的转置的转置是A
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
print(A.T.T == A)

# 2. 给两个矩阵，证明它们转置的和等于它们和的转置
B = A.clone()
C = A + B
print(A.T + B.T == C.T)

# 3. 给定任意方阵A，A+A^T总是对称吗
A = torch.arange(25,dtype=torch.float32).reshape(5,5)
print(A+A.T == (A+A.T).T)

# 4. 定义形状(2,3,4)的张量X，求len(X)的结果
X = torch.arange(24).reshape(2,3,4)
print(len(X)) # 2

# 5. 对于任意形状的张量X，len(X)是否总是对应于X特定轴的长度？这个轴是什么？：是的，总是第一维（最外层的长度)

# 6. A / A.sum(axis=1)的结果
print(A)
print(A / A.sum(axis=1)) # A的每一行的第i个数除以第i行求和的结果

# 7. 考虑一个具有形状(2,3,4)的张量X，在轴0，1，2上的求和输出是什么形状
print(X.sum(axis=0).shape,X.sum(axis=1).shape,X.sum(axis=2).shape)

# 8. 为linalg.norm函数（求范数）提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到什么？
A, B = torch.randn(2,3,4), torch.randn(3,4)
outputs1 = torch.linalg.norm(A)
outputs2 = torch.linalg.norm(B)
print(A)
print(outputs1)
print(B)
print(outputs2)