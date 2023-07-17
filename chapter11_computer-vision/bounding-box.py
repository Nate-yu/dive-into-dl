import torch
from d2l import torch as d2l
d2l.set_figsize()
img = d2l.plt.imread('./img/catdog.jpg')
d2l.plt.imshow(img)
# d2l.plt.show()

# 边界框
def box_corner_to_center(boxes):
    # 从（左上，右下）转换到（中间，宽度，高度）
    # boxes 是一个 Numpy 数组，其中每一行表示一个边界框, 包含左上角和右下角的坐标
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1) # torch.stack 函数将这些值沿着最后一个维度堆叠起来，得到一个形状为 (N, 4) 的张量，其中 N 是边界框的数量
    return boxes # 这个张量的每一行表示一个边界框的中心坐标、宽度和高度。函数返回这个张量作为结果

def box_center_to_corner(boxes):
    # 从（中间，宽度，高度）转换到（左上，右下）
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 根据坐标信息定义图像中狗和猫的边界框
# 图像中坐标的原点是图像的左上角，向右的方向为x轴的正方向，向下的方向为y轴的正方向
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

# 可以通过转换两次来验证边界框转换函数的正确性
boxes = torch.tensor((dog_bbox, cat_bbox))
# print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

# 我们可以将边界框在图中画出，以检查其是否准确
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy = (bbox[0],bbox[1]), width = bbox[2] - bbox[0], height = bbox[3] - bbox[1],
        fill = False, edgecolor = color, linewidth = 2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue')) # 向图像中添加一个形状
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()