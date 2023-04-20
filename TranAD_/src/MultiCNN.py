import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os

from src.CBlock import CBlock


class MultiCNN(nn.Module):

    def __init__(self, kernel_size = 96, d_model = 512 , n_heads = 8, depth = 1):
        super(MultiCNN, self).__init__()
        self.blocks = nn.ModuleList([
            CBlock(d_model=d_model) for i in range (depth)
        ])
        # self.CBlock = CBlock(d_model=d_model)
        # self.kernel_size = kernel_size
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.MBConv = MBConv(in_channels=512,out_channels=512)
        # self.DynamicCnn = Dynamic_conv1d(in_planes=d_model, out_planes=d_model, kernel_size=self.kernel_size, ratio=0.25, padding= 0, stride=1 )
        # self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.conv4 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.conv5 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.conv6 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.conv7 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.conv8 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel_size, stride=1)
        # self.out_projection = nn.Linear(d_model * n_heads, d_model)

    def forward(self, x, x1, x2,attn_mask=None):
        # 32 96 512
        # x = torch.cat((x,x[:,:-1,:]),dim=1)
        # x1 = self.conv1(x.permute(0, 2, 1)).transpose(1, 2)
        # x2 = self.conv2(x.permute(0, 2, 1)).transpose(1, 2)
        # x3 = self.conv3(x.permute(0, 2, 1)).transpose(1, 2)
        # x4 = self.conv4(x.permute(0, 2, 1)).transpose(1, 2)
        # x5 = self.conv5(x.permute(0, 2, 1)).transpose(1, 2)
        # x6 = self.conv6(x.permute(0, 2, 1)).transpose(1, 2)
        # x7 = self.conv7(x.permute(0, 2, 1)).transpose(1, 2)
        # x8 = self.conv8(x.permute(0, 2, 1)).transpose(1, 2)
        # x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),dim=2)
        # x = self.out_projection(x)
        for cblock in self.blocks:
            x = cblock(x)


        return x, None

if __name__ == '__main__':
    x = torch.rand(32,96,512)
    print(x.shape)
    Dy = MultiCNN()
    x,_ = Dy(x,x,x)
    print(x.shape)
