import torch

from data import dictionary_x, dictionary_y


def mask_pad(data: torch.tensor) -> torch.tensor:
    # b句话,每句话50个词,这里是还没embedding的
    # data=[b,50]
    # 判断每个词是不是<pad>
    mask = data == dictionary_x['<PAD>']
    # [b,50]->[b,1,1,50]
    mask = mask.reshape(-1, 1, 1, 50)

    # 在计算注意力时,是计算50个词和50个词相互之间的注意力,所以是个50*50的矩阵
    # pad的列是true,意味着任何对pad的注意力都是0
    # 但是pad本身对其他词的注意力并不是0
    # 所以pad行不是true

    # 复制n次
    # [b,1,1,50]->[b,1,50,50]
    mask = mask.expand(-1, 1, 50, 50)
    return mask


def mask_tril(data: torch.tensor):
    # b句话,每句话50个词,这里是还没embedding的
    # data=[b,50]
    # 50*50的矩阵表示每个词对其他词是否可见
    # 上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己和他之前的次,看不到后面的词
    # [1,50,50]
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))

    # 判断y当中每个词是不是pad,如果是pad则不可见
    # [b,50]
    mask = data == dictionary_y['<PAD>']

    # 变形+转型,为了之后的计算
    # [b,1,50]
    mask = mask.unsqueeze(1).long()

    # mask和tril求并集
    # [b,1,50]+[1,50,50]->[b,50,50]
    mask = mask + tril

    # 转bool型
    mask = mask > 0
    # 转bool型,增加一个维度,便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1)
    return mask
