import torch
import torchvision
X = torch.arange(16.).reshape(1,1,4,4)
print(X)

# 让我们进一步假设输入图像的高度和宽度都是40像素，且选择性搜索在此图像上生成了两个提议区域。 每个区域由5个元素表示：区域目标类别、左上角和右下角的(x,y)坐标
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
print(torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1))