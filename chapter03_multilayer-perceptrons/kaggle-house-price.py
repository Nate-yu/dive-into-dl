# 1. 下载和缓存数据集
import hashlib
import os
import tarfile
import zipfile
import requests

# 建立字典DATA_HUB
# 它可以将数据集名称的字符串映射到数据集相关的二元组上，这个二元组包含数据集的url和验证文件完整性的sha-1密钥。所有类似的数据集都托管在地址为DATA_URL的站点上。
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 下面的download函数用来下载数据集
# 将数据集缓存在本地目录（默认情况下为./data）中， 并返回下载文件的名称。 
# 如果缓存目录中已经存在此数据集文件，并且其sha-1与存储在DATA_HUB中的相匹配， 我们将使用缓存的文件，以避免重复的下载。
def download(name, cache_dir=os.path.join('.', 'data')):
    # 载一个DATA_HUB中的文件，返回本地文件名
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}" # 检查 name 是否在 DATA_HUB 中
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1]) # 将 cache_dir 和从 url 中提取的文件名拼接成一个完整的本地文件路径
    if os.path.exists(fname): # 避免重复下载
        sha1 = hashlib.sha1() # 创建一个 SHA-1 哈希对象
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576) # 读取 1048576 字节（1MB）的数据
                if not data: break
                sha1.update(data) # 更新 SHA-1 哈希对象
        if sha1.hexdigest() == sha1_hash: # 检查文件是否完整，并且SHA-1码对应
            return fname # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname # 成功下载并缓存文件名称，返回文件名称

# 我们还需实现两个实用函数：一个将下载并解压缩一个zip或tar文件， 另一个是将所有数据集从DATA_HUB下载到缓存目录中
def download_extract(name, folder=None):
    # 下载并解压zip/tar文件
    fname = download(name) # 返回文件名称
    base_dir = os.path.dirname(fname) # 获取文件名称所对应的目录名称
    data_dir, ext = os.path.splitext(fname) # 获取文件名称和所对应的文件扩展名称
    if ext == '.zip':
        fp = zipfile.ZipFile(fname) # 创建一个 zipfile 对象
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else: assert False # 只有zip/tar文件可以被解压缩
    fp.extractall(base_dir) # 解压缩文件夹内的所有文件到指定的目录中(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir # 返回解压后的文件夹或文件名称

def download_all():
    # 下载DATA_HUB中的所有文件
    for name in DATA_HUB:
        download(name)

