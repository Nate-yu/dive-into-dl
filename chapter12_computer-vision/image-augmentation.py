import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('./img/cat1.jpg')
""" d2l.plt.imshow(img)
d2l.plt.show() """

# 定义辅助函数apply, 此函数在输入图像img上多次运行图像增广方法aug并显示所有结果
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

# 翻转和裁剪
# 使用transforms模块来创建RandomFlipLeftRight实例，这样就各有50%的几率使图像向左或向右翻转
# apply(img, torchvision.transforms.RandomHorizontalFlip())

# 创建一个RandomFlipTopBottom实例，使图像各有50%的几率向上或向下翻转
# apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值
# 然后，区域的宽度和高度都被缩放到200像素
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200,200),scale=(0.1,1.0),ratio=(0.5,2.0)
)
# apply(img, shape_aug)

# 改变颜色
# 随机更改图像的亮度，随机值为原始图像的50%到150%之间
# apply(img, torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0))

# 随机更改图像的色调
# apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))

# 创建一个RandomColorJitter实例，并设置如何同时随机更改图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
col_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, col_aug)

# 结合多种图像增广方法
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),shape_aug,col_aug])
# apply(img, augs)

# 使用图像增广进行训练
all_images = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
# d2l.plt.show()

# 只使用最简单的随机左右翻转。 此外，使用ToTensor实例将一批图像转换为深度学习框架所要求的格式
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 定义一个辅助函数，以便于读取图像和应用图像增广
def load_cifa10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, 
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

# 多GPU训练
# 定义一个函数，使用多GPU对模型进行训练和评估
def train_batch_ch13(net,X,y,loss,trainer,devices):
    # 用多GPU进行小批量训练
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred,y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred,y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 
               devices=d2l.try_all_gpus()):
    # 用多GPU进行模型训练
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features,labels, loss, trainer, devices)
            metric.add(l,acc,labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
        f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
        f'{str(devices)}')
    
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

# 定义train_with_data_aug函数，使用图像增广来训练模型
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifa10(True, train_augs, batch_size)
    test_iter = load_cifa10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

train_with_data_aug(train_augs, test_augs, net)