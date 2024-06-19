import numpy as np
import random
import torch
from torch.utils.data import Dataset

# 定义词表, 词:数字
dictionary_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
dictionary_x = {word: i for i, word in enumerate(dictionary_x.split(','))}
dictionary_xr = [k for k, _ in dictionary_x.items()]
dictionary_y = {k.upper(): v for k, v in dictionary_x.items()}
dictionary_yr = [k for k, _ in dictionary_y.items()]


def get_data():
    # 定义词集合
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x',
        'c', 'v', 'b', 'n', 'm'
    ]

    # 定义每个词被选中的概率(模拟热门词和生僻词)
    p = np.array([i for i in range(1, 37)])
    p = p / p.sum()

    # 随机选n个词
    n = random.randint(30, 48)  # 句子长度
    x = np.random.choice(words, size=n, replace=True, p=p)
    x = x.tolist()

    # y是对x的变换得到的
    # 字母变大写,数字取10以内的互补数
    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)

    y = [f(i) for i in x]
    y = y + [y[-1]]
    # 逆序
    y = y[::-1]

    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    # 编码成数据
    x = [dictionary_x[i] for i in x]
    y = [dictionary_y[i] for i in y]

    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


class myDataset(Dataset):
    """
    定义数据集
    """

    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, item):
        return get_data()


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=myDataset(),
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)
