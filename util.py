import torch
import math


def attention(Q, K, V, mask):
    """
    注意力计算函数
    :param Q:
    :param K:
    :param V:
    :param mask:
    :return:
    """
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b,4,50,8]
    # [b,4,50,8]*[b,4,8,50]->[b,4,50,50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头维数的平方根,做数值缩放
    score = score / (8 ** 0.5)

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask=[b,1,50,50]
    score = score.masked_fill_(mask, -float('inf'))  # 把true替换成-inf
    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以v,得到最终的注意力结果
    # [b,4,50,50]*[b,4,50,8]->[b,4,50,8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b,4,50,8]->[b,50,32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)

    return score


class MultiHead(torch.nn.Module):
    """
    多头注意力计算层
    """

    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        self.out_fc = torch.nn.Linear(32, 32)
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量
        # Q,K,V=[b,50,32]
        b = Q.shape[0]

        # 保留下原始的Q,后面要做残差连接用
        clone_Q = Q.clone()

        # 规范化
        # 原论文中是吧归一化放在注意力之后,但实验证明,放在之前会有助于收敛
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变
        # [b,50,32]->[b,50,32]
        K = self.fc_K(K)
        Q = self.fc_Q(Q)
        V = self.fc_V(V)

        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b,50,32]->[b,4,50,8]
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        # 计算注意力
        # [b,4,50,8]->[b,50,32]
        score = attention(Q, K, V, mask)

        # 计算输出,纬度不变
        # [b,50,32]->[b,50,32]
        score = self.dropout(self.out_fc(score))

        # 残差连接
        score = clone_Q + score

        return score


class PositionEmbedding(torch.nn.Module):
    """
    位置编码层
    """

    def __init__(self):
        super().__init__()

        def get_pe(pos, i, d_model):
            """
            :param pos: 第几个词
            :param i: 第几个维度
            :param d_model: 维度总数
            :return:
            """
            fenmu = 1e5 ** (i / d_model)
            pe = pos / fenmu
            if i % 2 == 0:
                return math.sin(pe)
            else:
                return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层
        self.embed = torch.nn.Embedding(39, 32)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8,50]->[8,50,32]
        embed = self.embed(x)

        # 词编码和位置编码相加
        # [8,50,32]+[1,50,32]->[8,50,32]
        embed = embed + self.pe
        return embed


class FullyConnectedOutput(torch.nn.Module):
    """
    全连接输出层
    """

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1)
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面残差连接
        clone_x = x.clone()
        # 归一化
        x = self.norm(x)
        # 线性全连接层
        # [b,50,32]->[b,50,32]
        out = self.fc(x)

        out = out + clone_x
        return out
