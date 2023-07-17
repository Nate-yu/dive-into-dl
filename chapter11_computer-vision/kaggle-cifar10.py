import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 获取并组织数据集
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
# 如果使用完整的Kaggle竞赛的数据集，设置demo为False
demo = True
if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = './data/cifar-10/'

# 整理数据集
# 首先，我们读取CSV文件中的标签，它返回一个字典，
# 该字典将文件名中不带扩展名的部分映射到其标签
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行（列名）
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines] # l.rstrip()用于去除每行末尾的空格和换行符
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
""" print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values()))) """

# 将文件复制到目标目录
def copyfile(filename,target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

# 将验证集从原始的训练集中拆分出来, valid_ratio: 验证集在原始训练集中所占的比例
def reorg_train_valid(data_dir, labels, valid_ratio):
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1] # most_common()用于返回字典中最常见的元素及其计数。使用 [-1] 来获取最少的元素及其计数，使用 [1] 来获取计数值。
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio)) # math.floor向下取整
    label_count = {}
    # 遍历训练集中的每个文件
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        # 获取文件的标签
        label = labels[train_file.split('.')[0]]
        # 获取文件的路径
        fname = os.path.join(data_dir, 'train', train_file)
        # 将文件复制到train_valid_test/train_valid目录下的对应标签文件夹中
        copyfile(fname, os.path.join(data_dir, 'train_valid_test','train_valid', label))
        # 如果该标签的样本数小于n_valid_per_label，则将该文件复制到train_valid_test/valid目录下的对应标签文件夹中
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test','valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        # 否则将该文件复制到train_valid_test/train目录下的对应标签文件夹中
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test','train', label))
    return n_valid_per_label

# 在预测期间整理测试集，以方便读取
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))
        
# 调用前面定义的函数
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir,labels,valid_ratio)
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)

# 图像增广
transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 在测试期间，我们只对图像执行标准化，以消除评估结果中的随机性
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True,drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,drop_last=True)

# 定义模型
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")

# 定义训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    # 获取训练数据集的批次数
    num_batches, timer = len(train_iter), d2l.Timer()
    # 定义图例
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    # 定义动画
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    # 将模型放到多个GPU上进行训练
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # 开始训练
    for epoch in range(num_epochs):
        # 将模型设置为训练模式
        net.train()
        # 定义度量器
        metric = d2l.Accumulator(3)
        # 遍历训练数据集
        for i, (features, labels) in enumerate(train_iter):
            # 开始计时
            timer.start()
            # 训练一批次数据
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            # 更新度量器
            metric.add(l, acc, labels.shape[0])
            # 停止计时
            timer.stop()
            # 每训练完1/5个批次或者训练完最后一个批次，就更新动画
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        # 如果有验证数据集，就计算验证集的准确率并更新动画
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        # 更新学习率
        scheduler.step()
    # 计算训练结果
    measures = (f'train loss {metric[0] / metric[2]:.3f}, ' f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    # 输出训练结果
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}' f' examples/sec on {str(devices)}')

# 训练和验证模型
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4,5e-4
lr_period, lr_decay,net = 4, 0.9, get_net()
train(net,train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)