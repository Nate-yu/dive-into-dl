# 安装
## 在Anacoda中安装相应环境
打开Anacoda，进入`base`环境，在`base`环境中打开控制台输入以下命令。

1. 创建新环境`d2l`与安装python
```python
conda create --name d2l python=3.9 -y
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683686075798-f59a016a-0dd0-4d9c-97d0-226f4b8a276e.png#averageHue=%23191614&clientId=u71fd5555-854c-4&from=paste&height=95&id=u80d93854&originHeight=119&originWidth=775&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14828&status=done&style=none&taskId=uce920a46-c61d-4a1e-8fb8-1bdc3ef02bd&title=&width=620)

2. 进入新环境`d2l`
```python
conda activate d2l
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683686103825-64a8d7d2-62eb-415e-ad5c-5fbac4418440.png#averageHue=%23171513&clientId=u71fd5555-854c-4&from=paste&height=41&id=u4dc66a73&originHeight=51&originWidth=510&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=4289&status=done&style=none&taskId=u6c9229fd-a3b4-4b9e-8d16-b901d088f18&title=&width=408)

## 安装深度学习框架和d2l包

1. 安装`torch`与`torchvision`
```python
pip install torch==1.12.0
```
```python
pip install torchvision==0.13.0
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683686177171-89c331ff-1aa5-4802-ab41-3502480ccdc3.png#averageHue=%23131110&clientId=u71fd5555-854c-4&from=paste&height=694&id=ueaa8b38d&originHeight=868&originWidth=1901&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=177265&status=done&style=none&taskId=ua7eff285-d231-4556-8404-4ae208591f1&title=&width=1520.8)

2. 安装`d2l`包
```python
pip install d2l==0.17.6
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683686394426-3eb8f230-91e3-45a7-a546-41173f24b246.png#averageHue=%23141210&clientId=u71fd5555-854c-4&from=paste&height=94&id=u8f26d688&originHeight=117&originWidth=1011&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18585&status=done&style=none&taskId=u6b7af4fe-f429-45a9-a8c9-d335358bd9a&title=&width=809)

## 下载 D2L Notebook

1. 下载`d2l-zh`到指定路径

[https://zh-v2.d2l.ai/d2l-zh.zip](https://zh-v2.d2l.ai/d2l-zh.zip)

2. 进入此路径输入以下命令打开`JupyterNoteBook`
```python
jupyter notebook
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683687050637-99172950-bf2c-48b1-8236-a2a184c3550e.png#averageHue=%23161412&clientId=u71fd5555-854c-4&from=paste&height=513&id=ufbcf9541&originHeight=641&originWidth=1511&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=100351&status=done&style=none&taskId=u7b543999-8d5d-493a-adcd-18b9c7dc396&title=&width=1208.8)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683688539584-38769531-a2b6-472c-89d1-0e24535d5646.png#averageHue=%232b2e36&clientId=u71fd5555-854c-4&from=paste&height=394&id=u9b0dec0b&originHeight=493&originWidth=2489&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=51089&status=done&style=none&taskId=u5fa0eac7-e767-4114-a959-750ba836a6f&title=&width=1991.2)

3. 退出环境
```python
conda deactivate
```

# 1 补充预备知识
## 1.1 线性代数
### 1.1.1 张量的降维求和
对一个张量进行元素求和。
```python
x = torch.arange(4, dtype=torch.float32)
print(x,',',x.sum())
```
`tensor([0., 1., 2., 3.]) , tensor(6.)`

默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变成一个标量。我们还可以指定张量沿哪一个轴来通过求和降低维度。以矩阵为例，为了通过求和所有行的元素来降维（轴0），可以在调用函数时指定axis=0。 由于输入矩阵沿0轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失。

假设矩阵A为![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683774817794-5d4e8cfd-117c-465c-876b-802fc6c0e5b1.png#averageHue=%23262d33&clientId=uc2ec5947-6bd4-4&from=paste&height=84&id=u3176f75b&originHeight=105&originWidth=312&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7349&status=done&style=none&taskId=ud4605ff6-0cce-4d1a-a5a2-47359470c58&title=&width=250)
```python
A_sum_axis0 = A.sum(axis=0)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis0, A_sum_axis0.shape)
print(A_sum_axis1, A_sum_axis1.shape)
```
`tensor([40., 45., 50., 55.]) torch.Size([4])`<br />`tensor([ 6., 22., 38., 54., 70.]) torch.Size([5])`

同样，计算平均值的函数也可以沿指定轴降低张量的维度。
```python
print(A.mean(axis=0))
print(A.sum(axis=0)/A.shape[0])
```
`tensor([ 8.,  9., 10., 11.])`

### 1.1.2 非降维求和
有时在调用函数来计算总和或均值时保持轴数不变会很有用。
```python
# 非降维求和
sum_A = A.sum(axis=1, keepdim=True)
print(sum_A)
print(A / sum_A)
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683775191769-67556568-3d85-498c-ab65-f3ecc5609bae.png#averageHue=%23272e34&clientId=uc2ec5947-6bd4-4&from=paste&height=163&id=u87c897d3&originHeight=204&originWidth=437&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18259&status=done&style=none&taskId=ud21eebb3-c8f1-4fb9-9036-e8bf28c0216&title=&width=349.6)

如果我们想沿某个轴计算A元素的累积总和， 比如axis=0（按行计算），可以调用cumsum函数。 此函数不会沿任何轴降低输入张量的维度。
```python
A.cumsum(axis=0)
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683775230280-3386901f-c932-460d-8390-b8e29fee3103.png#averageHue=%23262d33&clientId=uc2ec5947-6bd4-4&from=paste&height=82&id=uc5c73727&originHeight=103&originWidth=335&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=6609&status=done&style=none&taskId=ub39d8515-38b1-4b03-a315-4c08bd1a5b6&title=&width=268)

### 1.1.3 一些乘法的使用
```python
# 点积
y = torch.ones(4,dtype=torch.float32)
print(x,y,torch.dot(x,y))

# 向量积
print(A.shape,x.shape,torch.mv(A,x)) # 使用torch.mv()函数时，A的列维数必须与x的维数相同

# 矩阵乘法
B = torch.ones(4,3)
print(A,B)
print(torch.mm(A,B))
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683775760909-6401e06f-3a98-4aba-9904-3cbabc27e64d.png#averageHue=%2323292f&clientId=uc2ec5947-6bd4-4&from=paste&height=268&id=ue56ff15d&originHeight=335&originWidth=765&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=32152&status=done&style=none&taskId=u55b7c17b-8684-4819-b2df-4a2719c9b4c&title=&width=612)

### 1.1.4 范数
> 非正式地说，向量的_范数_是表示一个向量有多大。 这里考虑的_大小_（size）概念不涉及维度，而是分量的大小。

在线性代数中，向量范数是将向量映射到标量的函数$f$。给定任意向量$\mathbf{x}$，向量范数要满足一些属性。

1. 如果我们按常数因子$\alpha$缩放向量的所有元素， 其范数也会按相同常数因子的绝对值缩放：$f(\alpha x) = |\alpha|f(x)$
2. 三角不等式：$f(x+y) \le f(x)+f(y)$
3. 范数必须是非负的：$f(x)\ge0$

范数听起来很像距离的度量。欧几里得距离和毕达哥拉斯定理中的非负性概念和三角不等式可能会给出一些启发。事实上，欧几里得距离是一个$L_2$范数：假设$n$维向量$\mathbf{x}$中的元素是$x_1,\ldots,x_n$，其$L_2$范数是向量元素平方和的平方根：$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$<br />其中，在$L_2$范数中常常省略下标$2$，也就是说$\|\mathbf{x}\|$等同于$\|\mathbf{x}\|_2$。<br />在代码中，我们可以按如下方式计算向量的$L_2$范数。
```python
# 范数
u = torch.tensor([3.0,-4.0])
print(torch.norm(u)) # 5
```

深度学习中更经常地使用$L_2$范数的平方，也会经常遇到$L_1$范数，它表示为向量元素的绝对值之和：$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|$<br />与$L_2$范数相比，$L_1$范数受异常值的影响较小。<br />为了计算$L_1$范数，我们将绝对值函数和按元素求和组合起来。
```python
print(torch.abs(u).sum()) # 7
```

$L_2$范数和$L_1$范数都是更一般的$L_p$范数的特例：$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}$

类似于向量的$L_2$范数，矩阵$\mathbf{X} \in \mathbb{R}^{m \times n}$的_Frobenius范数_（Frobenius norm）是矩阵元素平方和的平方根：<br />$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$

Frobenius范数满足向量范数的所有性质，它就像是矩阵形向量的$L_2$范数。<br />调用以下函数将计算矩阵的Frobenius范数。
```python
print(torch.norm(torch.ones((4,9)))) # 6
```

## 1.2 微积分
### 1.2.1 梯度
> 梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数。

我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的_梯度_（gradient）向量。<br />具体而言，设函数$f:\mathbb{R}^n\rightarrow\mathbb{R}$的输入是一个$n$维向量$\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，并且输出是一个标量。函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量：<br />$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$<br />其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在没有歧义时被$\nabla f(\mathbf{x})$取代。

# 5 深度学习计算
## 5.1 层和块
### 5.1.1 自定义块
在实现自定义块之前，我们简要总结一下每个块必须提供的基本功能。

1. 将输入数据作为其前向传播函数的参数。
2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
4. 存储和访问前向传播计算所需的参数。
5. 根据需要初始化模型参数。

### 5.1.2 顺序块
为了构建我们自己的简化的MySequential， 我们只需要定义两个关键函数：

1. 一种将块逐个追加到列表中的函数；
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

## 5.2 参数管理
### 5.2.2 参数初始化
> 默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵， 这个范围是根据输入和输出维度计算出的。 PyTorch的nn.init模块提供了多种预置初始化方法。

1. 内置初始化

让我们首先调用内置的初始化器。 下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0。
```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
```
```python
tensor([-0.0074,  0.0031, -0.0094,  0.0030]) tensor(0.)
```

我们还可以将所有参数初始化为给定的常数，比如初始化为1。
```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
```
```python
tensor([1., 1., 1., 1.]) tensor(0.)
```

我们还可以对某些块应用不同的初始化方法。 例如，下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。
```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```
```python
tensor([ 0.6557, -0.5915, -0.1561,  0.4981])
tensor([42., 42., 42., 42., 42., 42., 42., 42.])
```

2. 自定义初始化

有时，深度学习框架没有提供我们需要的初始化方法。在下面的例子中，我们使用以下的分布为任意权重参数$w$定义初始化方法：<br />$\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}$<br />同样，我们实现了一个my_init函数来应用到net。
```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])
```
```python
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[-0.0000,  8.0061,  0.0000,  6.0829],
        [ 0.0000, -0.0000,  7.8981, -6.5425]], grad_fn=<SliceBackward0>)
```

我们始终可以直接设置参数
```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])
```
```python
tensor([42.0000, 10.7377,  1.0000, -8.2703])
```

### 5.2.3 参数绑定
有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。
```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```
```python
tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])
```
这个例子表明第三个和第五个神经网络层的参数是绑定的。 它们不仅值相等，而且由相同的张量表示。 因此，如果我们改变其中一个参数，另一个参数也会改变。<br />这里有一个问题：当参数绑定时，梯度会发生什么情况？ 答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。

# 7 现代卷积神经网络
## 7.1 深度卷积神经网络（AlexNet）
AlexNet和LeNet的架构非常相似，如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686019079888-9c3944fd-38e2-4238-88b3-bd8bbda9c77d.png#averageHue=%23cddae4&clientId=ud9a33083-126c-4&from=paste&height=747&id=u00fa0c83&originHeight=996&originWidth=644&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=69923&status=done&style=none&taskId=u9b552b1f-0f30-4b9a-bf1c-c2c8c9fe13a&title=&width=483)<br />AlexNet和LeNet的设计理念非常相似，但也存在显著差异。

1. AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。
2. AlexNet使用ReLU而不是sigmoid作为其激活函数。

### 7.1.1 模型设计
在AlexNet的第一层，卷积窗口的形状是$11\times11$。由于ImageNet中大多数图像的宽和高比MNIST图像的多10倍以上，因此，需要一个更大的卷积窗口来捕获目标。第二层中的卷积窗口形状被缩减为$5\times5$，然后是$3\times3$。此外，在第一层、第二层和第五层卷积层之后，加入窗口形状为$3\times3$、步幅为2的最大汇聚层。而且，AlexNet的卷积通道数目是LeNet的10倍。

在最后一个卷积层后有两个全连接层，分别有4096个输出。这两个巨大的全连接层拥有将近1GB的模型参数。由于早期GPU显存有限，原版的AlexNet采用了双数据流设计，使得每个GPU只负责存储和计算模型的一半参数。幸运的是，现在GPU显存相对充裕，所以现在很少需要跨GPU分解模型。

### 7.1.2 激活函数
此外，AlexNet将sigmoid激活函数改为更简单的ReLU激活函数。 一方面，ReLU激活函数的计算更简单，它不需要如sigmoid激活函数那般复杂的求幂运算。 另一方面，当使用不同的参数初始化方法时，ReLU激活函数使训练模型更加容易。 当sigmoid激活函数的输出非常接近于0或1时，这些区域的梯度几乎为0，因此反向传播无法继续更新一些模型参数。 相反，ReLU激活函数在正区间的梯度总是1。 因此，如果模型参数没有正确初始化，sigmoid函数可能在正区间内得到几乎为0的梯度，从而使模型无法得到有效的训练。

### 7.1.3 容量控制和预处理
AlexNet通过暂退法（Dropout）控制全连接层的模型复杂度，而LeNet只使用了权重衰减。 为了进一步扩充数据，AlexNet在训练时增加了大量的图像增强数据，如翻转、裁切和变色。 这使得模型更健壮，更大的样本量有效地减少了过拟合。
```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))
```
	我们构造一个高度和宽度都为224的单通道数据，来观察每一层输出的形状。
```python
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```
```python
Conv2d output shape:         torch.Size([1, 96, 54, 54])
ReLU output shape:   torch.Size([1, 96, 54, 54])
MaxPool2d output shape:      torch.Size([1, 96, 26, 26])
Conv2d output shape:         torch.Size([1, 256, 26, 26])
ReLU output shape:   torch.Size([1, 256, 26, 26])
MaxPool2d output shape:      torch.Size([1, 256, 12, 12])
Conv2d output shape:         torch.Size([1, 384, 12, 12])
ReLU output shape:   torch.Size([1, 384, 12, 12])
Conv2d output shape:         torch.Size([1, 384, 12, 12])
ReLU output shape:   torch.Size([1, 384, 12, 12])
Conv2d output shape:         torch.Size([1, 256, 12, 12])
ReLU output shape:   torch.Size([1, 256, 12, 12])
MaxPool2d output shape:      torch.Size([1, 256, 5, 5])
Flatten output shape:        torch.Size([1, 6400])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 10])
```

### 7.1.4 读取数据集
尽管原文中AlexNet是在ImageNet上进行训练的，但在这里使用的是Fashion-MNIST数据集。因为即使在现代GPU上，训练ImageNet模型，同时使其收敛可能需要数小时或数天的时间。将AlexNet直接应用于Fashion-MNIST的一个问题是，Fashion-MNIST图像的分辨率（$28 \times 28$像素）低于ImageNet图像**。** 为了解决这个问题，我们将它们增加到$224 \times 224$（通常来讲这不是一个明智的做法，但在这里这样做是为了有效使用AlexNet架构）。这里需要使用`d2l.load_data_fashion_mnist`函数中的`resize`参数执行此调整。
```python
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

### 7.1.5 训练AlexNet
```python
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
![96e67b9fa02e4d293b154636a765319.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686019694485-f9009911-b6c9-44b0-8011-c925cf6f0d6e.png#averageHue=%23282522&clientId=ud9a33083-126c-4&from=paste&height=36&id=u1c758bcc&originHeight=45&originWidth=418&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=2736&status=done&style=none&taskId=ub04d26be-725b-44a9-89bf-d12f32557cd&title=&width=334.4)<br />![output_alexnet_180871_38_1.svg](https://cdn.nlark.com/yuque/0/2023/svg/25941432/1686019699393-649c1b2f-0fd2-4426-83f8-9f91afa665c1.svg#clientId=ud9a33083-126c-4&from=drop&id=ud089ff66&originHeight=305&originWidth=399&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=61084&status=done&style=none&taskId=uf6202f3d-faff-4ac3-841a-22f10a5d3bb&title=)

## 7.2 使用块的网络（VGG）
### 7.2.1 VGG块
经典卷积神经网络的基本组成部分是下面的这个序列：

1. 带填充以保持分辨率的卷积层；
2. 非线性激活函数，如ReLU；
3. 汇聚层，如最大汇聚层。

而一个VGG块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大池化层。在最初的VGG论文中 `Simonyan.Zisserman.2014`，作者使用了带有$3\times3$卷积核、填充为1（保持高度和宽度）的卷积层，和带有$2 \times 2$汇聚窗口、步幅为2（每个块后的分辨率减半）的最大池化层。在下面的代码中，我们定义了一个名为`vgg_block`的函数来实现一个VGG块。<br />该函数有三个参数，分别对应于卷积层的数量num_convs、输入通道的数量in_channels 和输出通道的数量out_channels.
```python
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

### 7.2.2 VGG网络
与AlexNet、LeNet一样，VGG网络可以分为两部分：第一部分主要由卷积层和汇聚层组成，第二部分由全连接层组成。如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686019909689-cda95139-0b1e-4530-9ec4-726a8b8f0b2e.png#averageHue=%23ebebeb&clientId=ud9a33083-126c-4&from=paste&height=475&id=u596f0c51&originHeight=594&originWidth=643&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=49790&status=done&style=none&taskId=ue122df9a-bd7b-46a8-a367-b2104953793&title=&width=514.4)<br />VGG神经网络连接上图的几个VGG块（在vgg_block函数中定义）。其中有超参数变量conv_arch。该变量指定了每个VGG块里卷积层个数和输出通道数。全连接模块则与AlexNet中的相同。

原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。
```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

下面的代码实现了VGG-11。可以通过在conv_arch上执行for循环来简单实现。
```python
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

接下来，我们将构建一个高度和宽度为224的单通道数据样本，以观察每个层输出的形状。
```python
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```
```python
Sequential output shape:     torch.Size([1, 64, 112, 112])
Sequential output shape:     torch.Size([1, 128, 56, 56])
Sequential output shape:     torch.Size([1, 256, 28, 28])
Sequential output shape:     torch.Size([1, 512, 14, 14])
Sequential output shape:     torch.Size([1, 512, 7, 7])
Flatten output shape:        torch.Size([1, 25088])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 10])
```
正如从代码中所看到的，我们在每个块的高度和宽度减半，最终高度和宽度都为7。最后再展平表示，送入全连接层处理。

### 7.2.3 训练模型
由于VGG-11比AlexNet计算量更大，因此我们构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集。
```python
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

除了使用略高的学习率外，模型训练过程与AlexNet类似。
```python
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
```python
loss 0.220, train acc 0.918, test acc 0.900
2578.4 examples/sec on cuda:0
```
![output_vgg_4a7574_71_1.svg](https://cdn.nlark.com/yuque/0/2023/svg/25941432/1686020088419-631f6558-0e76-4c39-9740-e34e465c74f7.svg#clientId=ud9a33083-126c-4&from=drop&id=ub40e763b&originHeight=305&originWidth=399&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=61066&status=done&style=none&taskId=ufe35186e-b262-44e7-ada6-4aef55d93eb&title=)

## 7.3 网络中的网络（NiN）
>  网络中的网络（NiN）提供了一个非常简单的解决方案：在每个像素的通道上分别使用多层感知机

### 7.3.1 NiN块
回想一下，卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本、通道、高度和宽度。另外，全连接层的输入和输出通常是分别对应于样本和特征的二维张量。NiN的想法是在每个像素位置（针对每个高度和宽度）应用一个全连接层。如果我们将权重连接到每个空间位置，我们可以将其视为$1\times 1$卷积层，或作为在每个像素位置上独立作用的全连接层。从另一个角度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）。

下图说明了VGG和NiN及它们的块之间主要架构差异。NiN块以一个普通卷积层开始，后面是两个$1 \times 1$的卷积层。这两个$1 \times 1$卷积层充当带有ReLU激活函数的逐像素全连接层。第一层的卷积窗口形状通常由用户设置。随后的卷积窗口形状固定为$1 \times 1$。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686105807334-fac6378b-7e11-4079-b1bd-de1c51093396.png#averageHue=%23ededed&clientId=udeb62791-e658-4&from=paste&height=698&id=bzEIu&originHeight=873&originWidth=944&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=64351&status=done&style=none&taskId=u1724977a-6d26-4ff1-902c-5ff8b139c21&title=&width=755.2)
```python
import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

### 7.3.2 NiN模型
最初的NiN网络是在AlexNet后不久提出的，显然从中得到了一些启示。NiN使用窗口形状为$11\times 11$、$5\times 5$和$3\times 3$的卷积层，输出通道数量与AlexNet中的相同。<br />每个NiN块后有一个最大池化层，汇聚窗口形状为$3\times 3$，步幅为2。

NiN和AlexNet之间的一个显著区别是NiN完全取消了全连接层。相反，NiN使用一个NiN块，其输出通道数等于标签类别的数量。最后放一个*全局平均汇聚层*（global average pooling layer），生成一个对数几率（logits）。NiN设计的一个优点是，它显著减少了模型所需参数的数量。然而，在实践中，这种设计有时会增加训练模型的时间。
```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
```

我们创建一个数据样本来查看每个块的输出形状。
```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686105983512-8645fde4-349f-4375-9b04-8eed129b74b7.png#averageHue=%232a2723&clientId=udeb62791-e658-4&from=paste&height=162&id=ub3ab911e&originHeight=203&originWidth=575&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=23219&status=done&style=none&taskId=ud3e8b7c6-b2c5-4284-8b60-801e9e74a8d&title=&width=460)

### 7.3.3 训练模型
和以前一样，我们使用Fashion-MNIST来训练模型。训练NiN与训练AlexNet、VGG时相似。
```python
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686106036597-fbd935a4-4430-421f-a60f-85e487960792.png#averageHue=%23292623&clientId=udeb62791-e658-4&from=paste&height=33&id=u99950207&originHeight=41&originWidth=411&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3250&status=done&style=none&taskId=u6c93a0b6-1127-4cc1-9808-7b34a5ab592&title=&width=328.8)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686105925286-aff49df3-bbe3-4201-ae9d-9fd3f95d6381.png#averageHue=%23f6f6f6&clientId=udeb62791-e658-4&from=paste&height=250&id=uc6ca2886&originHeight=312&originWidth=437&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24052&status=done&style=none&taskId=ud9c2e6a9-0899-4cc1-a0e2-b09d94a7d20&title=&width=349.6)

## 7.4 含并行连结的网络（GoogLeNet）
### 7.4.1 Inception块
在GoogLeNet中，基本的卷积块被称为_Inception块_（Inception block）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686190971029-cf70c2b7-4d90-42d6-91ce-1797dc290e2b.png#averageHue=%23eaeaea&clientId=u56fc2f0a-7892-4&from=paste&height=239&id=u71acea4b&originHeight=299&originWidth=851&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=25364&status=done&style=none&taskId=ub9be9cc1-4b7f-4807-bad1-cdf716982ed&title=&width=680.8)<br />如上图所示，Inception块由四条并行路径组成。前三条路径使用窗口大小为$1\times 1$、$3\times 3$和$5\times 5$的卷积层，从不同空间大小中提取信息。中间的两条路径在输入上执行$1\times 1$卷积，以减少通道数，从而降低模型的复杂性。第四条路径使用$3\times 3$最大池化层，然后使用$1\times 1$卷积层来改变通道数。这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。在Inception块中，通常调整的超参数是每层输出通道数。
```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
```

### 7.4.2 GoogLeNet模型
如下图所示，GoogLeNet一共使用9个Inception块和全局平均汇聚层的堆叠来生成其估计值。Inception块之间的最大汇聚层可降低维度。 第一个模块类似于AlexNet和LeNet，Inception块的组合从VGG继承，全局平均汇聚层避免了在最后使用全连接层。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686191134101-795d1c2e-ab6b-477a-afbc-84df9105f6fd.png#averageHue=%23e7e7e6&clientId=u56fc2f0a-7892-4&from=paste&height=618&id=ua06ae653&originHeight=772&originWidth=318&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=32254&status=done&style=none&taskId=u1f6a9908-b368-451d-ba50-f64ae98c3b4&title=&width=254.4)

现在，我们逐一实现GoogLeNet的每个模块。第一个模块使用64个通道、$7\times 7$卷积层。
```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第二个模块使用两个卷积层：第一个卷积层是64个通道、$1\times 1$卷积层；第二个卷积层使用将通道数量增加三倍的$3\times 3$卷积层。这对应于Inception块中的第二条路径。
```python
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第三个模块串联两个完整的Inception块。第一个Inception块的输出通道数为$64+128+32+32=256$，四个路径之间的输出通道数量比为$64:128:32:32=2:4:1:1$。第二个和第三个路径首先将输入通道的数量分别减少到$96/192=1/2$和$16/192=1/12$，然后连接第二个卷积层。第二个Inception块的输出通道数增加到$128+192+96+64=480$，四个路径之间的输出通道数量比为$128:192:96:64 = 4:6:3:2$。第二条和第三条路径首先将输入通道的数量分别减少到$128/256=1/2$和$32/256=1/8$。
```python
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第四模块更加复杂，它串联了5个Inception块，其输出通道数分别是$192+208+48+64=512$、$160+224+64+64=512$、$128+256+64+64=512$、$112+288+64+64=528$和$256+320+128+128=832$。这些路径的通道数分配和第三模块中的类似，首先是含$3×3$卷积层的第二条路径输出最多通道，其次是仅含$1×1$卷积层的第一条路径，之后是含$5×5$卷积层的第三条路径和含$3×3$最大汇聚层的第四条路径。其中第二、第三条路径都会先按比例减小通道数。这些比例在各个Inception块中都略有不同。
```python
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第五模块包含输出通道数为$256+320+128+128=832$和$384+384+128+128=1024$的两个Inception块。其中每条路径通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均汇聚层，将每个通道的高和宽变成1。最后我们将输出变成二维数组，再接上一个输出个数为标签类别数的全连接层。
```python
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

GoogLeNet模型的计算复杂，而且不如VGG那样便于修改通道数。 为了使Fashion-MNIST上的训练短小精悍，我们将输入的高和宽从224降到96，这简化了计算。下面演示各个模块输出的形状变化。
```python
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686191550346-d6103ee6-374a-4b12-8ab0-c299dc5eb935.png#averageHue=%23292623&clientId=u56fc2f0a-7892-4&from=paste&height=97&id=u94f0d921&originHeight=121&originWidth=585&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13243&status=done&style=none&taskId=ubb32c24f-ec23-457a-8eb6-366da6d4b09&title=&width=468)

### 7.4.3 训练模型
和以前一样，我们使用Fashion-MNIST数据集来训练我们的模型。在训练之前，我们将图片转换为$96 \times 96$分辨率。
```python
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686191610240-03482628-8e85-4049-a19a-a5a53ffa08e2.png#averageHue=%23272421&clientId=u56fc2f0a-7892-4&from=paste&height=36&id=u931f0bb7&originHeight=45&originWidth=430&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3503&status=done&style=none&taskId=uac7a9e7c-daca-4471-9b01-29f366d6c53&title=&width=344)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686190885056-ac239f45-1a4a-4670-8613-3879df8d3336.png#averageHue=%23f6f6f6&clientId=u56fc2f0a-7892-4&from=paste&height=250&id=ud878f7c5&originHeight=312&originWidth=437&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=22757&status=done&style=none&taskId=u7fb721e9-e3d3-4414-b3a7-3c40431b8be&title=&width=349.6)

## 7.5 残差网络（ResNet）
### 7.5.1 函数类
首先，假设有一类特定的神经网络架构$\mathcal{F}$，它包括学习速率和其他超参数设置。对于所有$f \in \mathcal{F}$，存在一些参数集（例如权重和偏置），这些参数可以通过在合适的数据集上进行训练而获得。现在假设$f^*$是我们真正想要找到的函数，如果是$f^* \in \mathcal{F}$，那我们可以轻而易举的训练得到它，但通常我们不会那么幸运。相反，我们将尝试找到一个函数$f^*_\mathcal{F}$，这是我们在$\mathcal{F}$中的最佳选择。例如，给定一个具有$\mathbf{X}$特性和$\mathbf{y}$标签的数据集，我们可以尝试通过解决以下优化问题来找到它：<br />$f^*_\mathcal{F} := \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$

那么，怎样得到更近似真正$f^*$的函数呢？唯一合理的可能性是，我们需要设计一个更强大的架构$\mathcal{F}'$。换句话说，我们预计$f^*_{\mathcal{F}'}$比$f^*_{\mathcal{F}}$“更近似”。然而，如果$\mathcal{F} \not\subseteq \mathcal{F}'$，则无法保证新的体系“更近似”。<br />事实上，$f^*_{\mathcal{F}'}$可能更糟：如下图所示，对于非嵌套函数（non-nested function）类，较复杂的函数类并不总是向“真”函数$f^*$靠拢（复杂度由$\mathcal{F}_1$向$\mathcal{F}_6$递增）。在下图的左边，虽然$\mathcal{F}_3$比$\mathcal{F}_1$更接近$f^*$，但$\mathcal{F}_6$却离的更远了。相反对于下图右侧的嵌套函数（nested function）类$\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$，我们可以避免上述问题。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686364287566-7a230360-c77f-47fe-8a6b-ddc23ebbeeb6.png#averageHue=%23d0d0d0&clientId=u1ec59fa7-6cfb-4&from=paste&height=389&id=u4daf9db5&originHeight=486&originWidth=1107&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=56262&status=done&style=none&taskId=u6667d7ff-2d10-4500-bbd9-ff1c8935cbf&title=&width=885.6)

因此，只有当较复杂的函数类包含较小的函数类时，我们才能确保提高它们的性能。对于深度神经网络，如果我们能将新添加的层训练成_恒等映射_（identity function）$f(\mathbf{x}) = \mathbf{x}$，新模型和原模型将同样有效。同时，由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差。

针对这一问题，何恺明等人提出了_残差网络_（ResNet）。它在2015年的ImageNet图像识别挑战赛夺魁，并深刻影响了后来的深度神经网络的设计。残差网络的核心思想是：每个附加层都应该更容易地包含原始函数作为其元素之一。于是，_残差块_（residual blocks）便诞生了，这个设计对如何建立深层神经网络产生了深远的影响。

### 7.5.2 残差块
让我们聚焦于神经网络局部：如下图所示，假设我们的原始输入为$x$，而希望学出的理想映射为$f(\mathbf{x})$（ 作为下图上方激活函数的输入）。下图中左图虚线框中的部分需要直接拟合出该映射$f(\mathbf{x})$，而右图虚线框中的部分则需要拟合出残差映射$f(\mathbf{x}) - \mathbf{x}$。残差映射在现实中往往更容易优化。以本节开头提到的恒等映射作为我们希望学出的理想映射$f(\mathbf{x})$，我们只需将下图中右图虚线框内上方的加权运算（如仿射）的权重和偏置参数设成0，那么$f(\mathbf{x})$即为恒等映射。实际中，当理想映射$f(\mathbf{x})$极接近于恒等映射时，残差映射也易于捕捉恒等映射的细微波动。下图中右图是ResNet的基础架构--_残差块_（residual block）。在残差块中，输入可通过跨层数据线路更快地向前传播。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686364574858-c3efec75-1f17-4c08-8ead-3fffee82fcfd.png#averageHue=%23ededed&clientId=u1ec59fa7-6cfb-4&from=paste&height=498&id=uea11bd5d&originHeight=623&originWidth=790&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=29226&status=done&style=none&taskId=u2fb5bb25-1b21-4ac7-885b-a0c88dcadf1&title=&width=632)

ResNet沿用了VGG完整的$3\times 3$卷积层设计。残差块里首先有2个有相同输出通道数的$3\times 3$卷积层。每个卷积层后接一个批量规范化层和ReLU激活函数。然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前。这样的设计要求2个卷积层的输出与输入形状一样，从而使它们可以相加。如果想改变通道数，就需要引入一个额外的$1\times 1$卷积层来将输入变换成需要的形状后再做相加运算。残差块的实现如下：
```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

如下图所示，此代码生成两种类型的网络： 一种是当`use_1x1conv=False`时，应用ReLU非线性函数之前，将输入添加到输出。 另一种是当`use_1x1conv=True`时，添加通过1×1卷积调整通道和分辨率。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686364679290-bae08ee4-711e-4527-aedd-8b230eee0252.png#averageHue=%23ededed&clientId=u1ec59fa7-6cfb-4&from=paste&height=585&id=u193a3aca&originHeight=731&originWidth=1060&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=40297&status=done&style=none&taskId=ube9e35fe-2446-489e-9242-49da0bdd49b&title=&width=848)

下面我们来查看输入和输出形状一致的情况。
```python
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)
```
```python
torch.Size([4, 3, 6, 6])
```

我们也可以在增加输出通道数的同时，减半输出的高和宽。
```python
blk = Residual(3,6, use_1x1conv=True, strides=2)
print(blk(X).shape)
```

### 7.5.3 ResNet模型
ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道数为64、步幅为2的$7 \times 7$卷积层后，接步幅为2的$3 \times 3$的最大汇聚层。不同之处在于ResNet每个卷积层后增加了批量规范化层。
```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

GoogLeNet在后面接了4个由Inception块组成的模块。 ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。 第一个模块的通道数同输入通道数一致。 由于之前已经使用了步幅为2的最大汇聚层，所以无须减小高和宽。 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。<br />下面我们来实现这个模块。注意，我们对第一个模块做了特别处理。
```python
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```
接着在ResNet加入所有残差块，这里每个模块使用2个残差块。
```python
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

最后，与GoogLeNet一样，在ResNet中加入全局平均汇聚层，以及全连接层输出。
```python
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

每个模块有4个卷积层（不包括恒等映射的$1\times 1$卷积层）。加上第一个$7\times 7$卷积层和最后一个全连接层，共有18层。因此，这种模型通常被称为ResNet-18。通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。虽然ResNet的主体架构跟GoogLeNet类似，但ResNet架构更简单，修改也更方便。这些因素都导致了ResNet迅速被广泛使用。下图描述了完整的ResNet-18。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686364976819-378365d1-bf36-4d99-b04f-6f08b2c98598.png#averageHue=%23e8e8e8&clientId=u1ec59fa7-6cfb-4&from=paste&height=1009&id=uf4bdec8a&originHeight=1261&originWidth=346&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=33331&status=done&style=none&taskId=u038fe514-6535-49e2-9f83-009efe03c16&title=&width=276.8)<br />在训练ResNet之前，让我们观察一下ResNet中不同模块的输入形状是如何变化的。 在之前所有架构中，分辨率降低，通道数量增加，直到全局平均汇聚层聚集所有特征。
```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```
```python
Sequential output shape:     torch.Size([1, 64, 56, 56])
Sequential output shape:     torch.Size([1, 64, 56, 56])
Sequential output shape:     torch.Size([1, 128, 28, 28])
Sequential output shape:     torch.Size([1, 256, 14, 14])
Sequential output shape:     torch.Size([1, 512, 7, 7])
AdaptiveAvgPool2d output shape:      torch.Size([1, 512, 1, 1])
Flatten output shape:        torch.Size([1, 512])
Linear output shape:         torch.Size([1, 10])
```

### 7.5.4 训练模型
同之前一样，我们在Fashion-MNIST数据集上训练ResNet。
```python
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686363267448-17e1d231-32f5-4cec-8f12-f0820dc799ed.png#averageHue=%23262421&clientId=u1ec59fa7-6cfb-4&from=paste&height=34&id=u70e5c138&originHeight=42&originWidth=409&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=6075&status=done&style=none&taskId=ucdb582de-170f-42e9-90b5-7ca853d5921&title=&width=327.2)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1686363238986-016a6cd5-2337-4aa4-9de5-0378eb74755b.png#averageHue=%23f6f6f6&clientId=u1ec59fa7-6cfb-4&from=paste&height=250&id=u48a48dc0&originHeight=312&originWidth=437&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24250&status=done&style=none&taskId=u50627ea3-9fb9-47ab-b10e-26667eb3698&title=&width=349.6)
