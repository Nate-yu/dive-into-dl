import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 阅读内容与图像风格
d2l.set_figsize()
content_img = d2l.Image.open('./img/rainier.jpg')
style_img = d2l.Image.open('./img/autumn-oak.jpg')
""" d2l.plt.imshow(content_img)
d2l.plt.show()
d2l.plt.imshow(style_img)
d2l.plt.show() """

# 预处理和后处理
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1,2,0) * rgb_std + rgb_mean, 0,1)
    return torchvision.transforms.ToPILImage()(img.permute(2,0,1))

# 抽取图像特征
pretrained_net = torchvision.models.vgg19(pretrained=True)
style_layers, content_layers = [0, 5, 10, 19, 28], [25]