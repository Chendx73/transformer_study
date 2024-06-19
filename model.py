import torch
from util import (
    MultiHead,
    FullyConnectedOutput,
    PositionEmbedding
)
from mask import mask_pad, mask_tril


class EncoderLayer(torch.nn.Module):
    """
    编码器层
    """

    def __init__(self):
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,维度不变
        # [b,50,32]->[b,50,32]
        score = self.mh(x, x, x, mask)
        # 全连接输出
        # [b,50,32]->[b,50,32]
        out = self.fc(score)
        return out


class Encoder(torch.nn.Module):
    """
    编码器
    """

    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


class DecoderLayer(torch.nn.Module):
    """
    解码器层
    """

    def __init__(self):
        super().__init__()
        self.mh1 = MultiHead()
        self.mh2 = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算 y的自注意力,维度不变
        # [b,50,32]->[b,50,32]
        y = self.mh1(y, y, y, mask_tril_y)
        # 结合x和y的注意力计算,维度不变
        # [b,50,32],[b,50,32]->[b,50,32]
        y = self.mh2(y, x, x, mask_pad_x)
        # 全连接输出,维度不变
        # [b,50,32]->[b,50,32]
        y = self.fc(y)
        return y


class Decoder(torch.nn.Module):
    """
    解码器
    """

    def __init__(self):
        super().__init__()
        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 39)

    def forward(self, x: torch.tensor, y: torch.tensor):
        # [b,1,50,50]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)
        # 编码,添加位置信息
        # x=[b,50]->[b,50,32]
        # y=[b,50]->[b,50,32]
        x, y = self.embed(x), self.embed(y)

        # 编码层计算
        # [b,50,32]->[b,50,32]
        x = self.encoder(x, mask_pad_x)

        # 解码层计算
        # [b,50,32],[b,50,32]->[b,50,32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        # 全连接输出
        y = self.fc_out(y)
        return y
