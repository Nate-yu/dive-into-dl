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
