import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SegNet import *
from model.ESSA import ESSA

''' SegNet with ESSA module '''


class SegNet_ESSA(nn.Module):

    def __init__(self, n_class:int, in_channels=4):
        super().__init__()

        self.in_channels = in_channels
        
        chs = [32,64,64,128,128]

        self.C1_ESSA = ESSA(32,  64, 1)
        self.C3_ESSA = ESSA(64, 128, 1)

        self.down1 = nn.Sequential(
            ConvBnRelu2d(in_channels, chs[0]),
            ConvBnRelu2d(chs[0], chs[0]),
        )
        self.down2 = nn.Sequential(
            ConvBnRelu2d(chs[0], chs[1]),
            ConvBnRelu2d(chs[1], chs[1]),
        )
        self.down3 = nn.Sequential(
            ConvBnRelu2d(chs[1], chs[2]),
            ConvBnRelu2d(chs[2], chs[2]),
            ConvBnRelu2d(chs[2], chs[2])
        )
        self.down4 = nn.Sequential(
            ConvBnRelu2d(chs[2], chs[3]),
            ConvBnRelu2d(chs[3], chs[3]),
            ConvBnRelu2d(chs[3], chs[3])
        )
        self.down5 = nn.Sequential(
            ConvBnRelu2d(chs[3], chs[4]),
            ConvBnRelu2d(chs[4], chs[4]),
            ConvBnRelu2d(chs[4], chs[4])
        )
        self.up5 = nn.Sequential(
            ConvBnRelu2d(chs[4], chs[4]),
            ConvBnRelu2d(chs[4], chs[4]),
            ConvBnRelu2d(chs[4], chs[3])
        )
        self.up4 = nn.Sequential(
            ConvBnRelu2d(chs[3], chs[3]),
            ConvBnRelu2d(chs[3], chs[3]),
            ConvBnRelu2d(chs[3], chs[2])
        )
        self.up3 = nn.Sequential(
            ConvBnRelu2d(chs[2]*2, chs[2]),
            ConvBnRelu2d(chs[2], chs[2]),
            ConvBnRelu2d(chs[2], chs[1])
        )
        self.up2 = nn.Sequential(
            ConvBnRelu2d(chs[1], chs[1]),
            ConvBnRelu2d(chs[1], chs[0])
        )
        self.up1 = nn.Sequential(
            ConvBnRelu2d(chs[0]*2, chs[0]),
            ConvBnRelu2d(chs[0], n_class)
        )

    def forward(self, x):
        x       = self.down1(x)
        x_p1    = self.C1_ESSA(x)
        x, ind1 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down2(x)
        x, ind2 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down3(x)
        x_p3    = self.C3_ESSA(x)
        x, ind3 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down4(x)
        x, ind4 = F.max_pool2d(x, 2, 2, return_indices=True)
        x       = self.down5(x)
        x, ind5 = F.max_pool2d(x, 2, 2, return_indices=True)

        x       = F.max_unpool2d(x, ind5, 2, 2)
        x       = self.up5(x)
        x       = F.max_unpool2d(x, ind4, 2, 2)
        x       = self.up4(x)
        x       = F.max_unpool2d(x, ind3, 2, 2)
        x       = self.up3(torch.cat([x, x_p3], dim=1))
        x       = F.max_unpool2d(x, ind2, 2, 2)
        x       = self.up2(x)
        x       = F.max_unpool2d(x, ind1, 2, 2)
        x       = self.up1(torch.cat([x, x_p1], dim=1))

        return x


if __name__ == '__main__':
    x = torch.rand([1, 4, 480, 640])
    model = SegNet_ESSA(n_class=9)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (1, 9, 480, 640), f'invalid shape: {y.shape}'

    print('[param_cnt]')
    print('  SegNet:',      sum(p.numel() for p in SegNet     (n_class=9).parameters()))
    print('  SegNet_ESSA:', sum(p.numel() for p in SegNet_ESSA(n_class=9).parameters()))

    print('Done!')
