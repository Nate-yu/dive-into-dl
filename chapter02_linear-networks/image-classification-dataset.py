import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 读取数据集
trans = transforms.ToTensor() # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
# print(len(mnist_train),len(mnist_test))
# print(mnist_train[0][0].shape)

# 返回Fashion-MNIST数据集的文本标签，用于在数字标签索引及其文本名称之间进行转换
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 创建一个函数来可视化这些样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): # scale是图片缩放比例
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)): # enumerate()函数将一个可迭代对象转换为一个枚举对象，同时返回每个元素的索引和值；zip()函数将imgs和axes打包成一个元组序列
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imgshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# 以下是训练数据集中前几个样本的图像及其相应的标签
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# d2l.plt.show()

# 读取小批量

batch_size = 256

# 使用4个进程来读取数据
def get_dataloader_workers():
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

# 查看读取训练数据所需时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
# print(f'{timer.stop():.2f} sec')

# 整合所有组件
# 定义load_data_fashion_mnist函数，用于获取和读取Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):
    # 下载Fashion-MNIST数据集，然后将其加载到内存中
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), 
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break