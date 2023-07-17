import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 获取数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

# 创建两个实例来分别读取训练和测试数据集中的所有图像文件
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 显示前8个正类样本图片和最后8张负类样本图片
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
""" d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
d2l.plt.show() """

# 数据增强
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256,256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# 定义和初始化模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# print(pretrained_net.fc)

# 目标模型finetune_net中成员变量features的参数被初始化为源模型相应层的模型参数
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features,2) # 将模型的最后一层全连接层的输出改为2
nn.init.xavier_uniform_(finetune_net.fc.weight) # 对新的全连接层的权重进行初始化 

# 微调模型
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size = 128, num_opochs = 5, param_group = True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'),transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devives = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params':params_1x},
                                   {'params':net.fc.parameters(),'lr':learning_rate*10}],
                                   lr = learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr = learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_opochs, devives)

# 使用较小的学习率，通过微调预训练获得的模型参数
train_fine_tuning(finetune_net,5e-5)