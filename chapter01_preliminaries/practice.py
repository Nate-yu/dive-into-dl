# 数据准备
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
print(y)
