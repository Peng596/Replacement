from torch.nn import Module
import torch
import torch.nn as nn

from module.feedForward import FeedForward
from module.multiHeadAttention import MultiHeadAttention
from module.MultiCNN import MultiCNN

class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.attn = nn.Conv1d(d_model, d_model, 9, padding=4, groups=d_model)

        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)
        self.MultiCNN = MultiCNN(d_model = d_model)
    def forward(self, x, stage):

        residual = x
        # x, score = self.MHA(x, stage)
        x , score = self.attn(x.transpose(1,2)).transpose(1,2) , None
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)
        # x ,score = MultiCNN(x,x,x)
        return x, score
