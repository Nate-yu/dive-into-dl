# 1. 下载和缓存数据集
import hashlib
import os
import tarfile
import zipfile
import requests

# 建立字典DATA_HUB
# 它可以将数据集名称的字符串映射到数据集相关的二元组上，这个二元组包含数据集的url和验证文件完整性的sha-1密钥。所有类似的数据集都托管在地址为DATA_URL的站点上。
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 下面的download函数用来下载数据集
# 将数据集缓存在本地目录（默认情况下为./data）中， 并返回下载文件的名称。 
# 如果缓存目录中已经存在此数据集文件，并且其sha-1与存储在DATA_HUB中的相匹配， 我们将使用缓存的文件，以避免重复的下载。
def download(name, cache_dir=os.path.join('.', 'data')):
    # 载一个DATA_HUB中的文件，返回本地文件名
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}" # 检查 name 是否在 DATA_HUB 中
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1]) # 将 cache_dir 和从 url 中提取的文件名拼接成一个完整的本地文件路径
    if os.path.exists(fname): # 避免重复下载
        sha1 = hashlib.sha1() # 创建一个 SHA-1 哈希对象
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576) # 读取 1048576 字节（1MB）的数据
                if not data: break
                sha1.update(data) # 更新 SHA-1 哈希对象
        if sha1.hexdigest() == sha1_hash: # 检查文件是否完整，并且SHA-1码对应
            return fname # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname # 成功下载并缓存文件名称，返回文件名称

# 我们还需实现两个实用函数：一个将下载并解压缩一个zip或tar文件， 另一个是将所有数据集从DATA_HUB下载到缓存目录中
def download_extract(name, folder=None):
    # 下载并解压zip/tar文件
    fname = download(name) # 返回文件名称
    base_dir = os.path.dirname(fname) # 获取文件名称所对应的目录名称
    data_dir, ext = os.path.splitext(fname) # 获取文件名称和所对应的文件扩展名称
    if ext == '.zip':
        fp = zipfile.ZipFile(fname) # 创建一个 zipfile 对象
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else: assert False # 只有zip/tar文件可以被解压缩
    fp.extractall(base_dir) # 解压缩文件夹内的所有文件到指定的目录中(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir # 返回解压后的文件夹或文件名称

def download_all():
    # 下载DATA_HUB中的所有文件
    for name in DATA_HUB:
        download(name)

# 2. 访问和读取数据集
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# 可以使用上面定义的脚本下载并缓存Kaggle房屋数据集
DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 使用pandas分别加载包含训练数据和测试数据的两个CSV文件
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 训练数据集包括1460个样本，每个样本80个特征和1个标签， 而测试数据集包含1459个样本，每个样本80个特征
""" print(train_data.shape)
print(test_data.shape) """

# 看看前四个和最后两个特征，以及相应标签（房价）
""" print(train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]]) """

# 在将数据提供给模型之前，将Id从数据集中删除
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 3. 数据预处理
""" 
1. 首先，我们将所有缺失的值替换为相应特征的平均值
2. 然后，为了将所有特征放在一个共同的尺度上， 我们通过将特征重新缩放到零均值和单位方差来标准化数据
3. 接下来，处理离散值
4.  最后，通过values属性，我们可以从pandas格式中提取NumPy格式，并将其转换为张量表示用于训练。
"""
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std())) # mean()与std()分别表示均值与标准差

# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
""" print(all_features.shape) """

n_train = train_data.shape[0] # 获取训练数据的行数（样本数量）
# 将预处理后的数据转换为PyTorch张量
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features =  torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 4. 训练
loss = nn.MSELoss()
in_features = train_features.shape[1]  # 输入特征数目

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# 训练函数将借助Adam优化器
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 使用Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step() # 更新参数
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None: # 验证集不为空的情况下计算测试误差
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# 5. K折交叉验证
def get_k_fold_dat(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k # 计算每个折的大小
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size) # 获取第j个折的索引
        X_part, y_part = X[idx,:], y[idx] # 将索引的元素存储到新的列表中，以便后面的x/y_part可以使用它们的末尾元素作为标志
        if j == i: # 如果j == i，则将X_part和y_part作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None: # 否则，将X_part和y_part添加到训练集
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
# 当我们在K折交叉验证中训练K次后，返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_dat(k,i,X_train,y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], 
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, ' f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# 6. 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 10, 0, 64
""" train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证：平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}') """

# 7. 提交Kaggle预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse: {float(train_ls[-1]):f}')
     # 将网络应用于测试集
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,num_epochs, lr, weight_decay, batch_size)
