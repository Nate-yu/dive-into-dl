import os
import torch
import torchvision
from d2l import torch as d2l

# 下载数据集
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar', '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

# 将所有输入的图像和标签读入内存
def read_voc_images(voc_dir, is_train=True):
    # 读取所有VOC图像并标注
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)

# 绘制前5个输入图像及其标签
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
""" d2l.show_images(imgs,2,n)
d2l.plt.show() """

# 列举RGB颜色值和类名
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 构建从RGB到VOC类别索引的映射
def voc_colormap2label():
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

# 将VOC标签中的RGB值映射到它们的类别索引
def voc_label_indices(colormap, colormap2label):
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], voc_colormap2label())
# print(y[105:115, 130:140], VOC_CLASSES[1])

# 预处理数据
def voc_rand_crop(feature, label, height, width):
    # 随机裁剪特征和标签图像
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1,2,0) for img in imgs]
""" d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
d2l.plt.show() """

# 自定义语义分割数据集类
class VOCSegDataset(torch.utils.data.Dataset):
    # 一个用于加载VOC数据集的自定义数据集
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # 用于图像标准化的转换
        self.crop_size = crop_size # 裁剪图像的大小
        features, labels = read_voc_images(voc_dir, is_train=is_train) # features: 标准化和过滤后的特征图像列表; labels: 过滤后的标签图像列表
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label() # 颜色映射到标签的字典
        print('read ' + str(len(self.features)) + ' examples')

    # 将图像数据标准化到0-1之间
    def normalize_image(self, img):
        return self.transform(img.float() / 255)
    
    # 用于过滤掉尺寸小于crop_size的图像
    def filter(self, imgs):
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and 
                                        img.shape[2] >= self.crop_size[1])]
    
    # 用于获取指定索引的特征图像和标签图像。在获取时，会对图像进行随机裁剪，并将标签图像的颜色映射到标签
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))
    
    # 用于获取特征图像的数量
    def __len__(self):
        return len(self.features)
    
# 读取数据集
crop_size = (320,480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)

batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, drop_last=True, num_workers=d2l.get_dataloader_workers())
""" for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break """

# 整合所有组件
# 下载并读取Pascal VOC2012语义分割数据集，返回训练集和测试集的数据迭代器
def load_data_voc(batch_size, crop_size):
    # 加载VOC语义分割数据集
    voc_dir = d2l.download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir), batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir), batch_size, drop_last=True, num_workers=num_workers)
    return train_iter, test_iter