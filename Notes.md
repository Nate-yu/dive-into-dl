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

