import os

# 创建一个人工数据集
os.makedirs(os.path.join('data'), exist_ok=True)
data_file = os.path.join('data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名：房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 从创建的CSV文件中加载原始数据集
import pandas as pd
data = pd.read_csv(data_file)
print(data)

# 处理缺失值
inputs, outputs = data.iloc[:,0:2], data.iloc[:,2] # iloc表示位置索引
inputs = inputs.fillna(inputs.mean()) # 将数值类型的NaN填充为平均值（3.0）
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True) # 对于inputs中的类别值或离散值，我们将“NaN”视为一个类别
print(inputs)

# 转换为张量格式
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X,'\n',y)