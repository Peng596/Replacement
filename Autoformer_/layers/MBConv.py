import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion=4, padding = 1, downsample=False):
        super(MBConv, self).__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(in_channels * expansion)
        if self.downsample:
            self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
            self.proj = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = padding, bias=False)
        self.conv = nn.Sequential(
            #Norm
            nn.BatchNorm1d(in_channels),
            #Narrow->Wide
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, padding = padding , stride=stride,),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            # DW
            #Wide -> Wide
            nn.Conv1d(in_channels = hidden_dim, out_channels= hidden_dim, groups= hidden_dim,kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            #Wide -> Narrow
            nn.Conv1d(in_channels = hidden_dim, out_channels= out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        #32 96 512
        if self.downsample:
            x = self.proj(self.pool(x.permute(0, 2, 1))).transpose(1, 2) + self.conv(x.permute(0, 2, 1)).transpose(1, 2)

        else:
            x = x + self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


if __name__ == '__main__':
    x= torch.rand((32,96,512))
    MB = MBConv(in_channels=512,out_channels=512,downsample=False)
    x = MB(x)
    print(x.shape)