import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MFNet import *
from model.ESSA import ESSA

''' MFNet with ESSA module '''


class MFNet_ESSA(nn.Module):

    def __init__(self, n_class:int):
        super().__init__()

        rgb_ch = [16,48,48,96,96]
        inf_ch = [16,16,16,36,36]

        self.C1_ESSA_rgb = ESSA(16, 32, 1)
        self.C3_ESSA_rgb = ESSA(48, 96, 1)
        self.C1_ESSA_inf = ESSA(16, 32, 1)
        self.C3_ESSA_inf = ESSA(16, 32, 1)

        self.conv1_rgb   = ConvBnLeakyRelu2d(3, rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb   = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb   = MiniInception(rgb_ch[3], rgb_ch[4])

        self.conv1_inf   = ConvBnLeakyRelu2d(1, inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf   = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf   = MiniInception(inf_ch[3], inf_ch[4])

        self.decode4     = ConvBnLeakyRelu2d(rgb_ch[3]+inf_ch[3], rgb_ch[2]+inf_ch[2])
        self.decode3     = ConvBnLeakyRelu2d((rgb_ch[2]+inf_ch[2])*2, rgb_ch[1]+inf_ch[1])
        self.decode2     = ConvBnLeakyRelu2d(rgb_ch[1]+inf_ch[1], rgb_ch[0]+inf_ch[0])
        self.decode1     = ConvBnLeakyRelu2d((rgb_ch[0]+inf_ch[0])*2, n_class)
        
    def forward(self, x):
        # split data into RGB and INF
        x_rgb = x[:,:3]
        x_inf = x[:,3:]
        
        # encode
        x_rgb    = self.conv1_rgb(x_rgb)
        E_rgb_p1 = self.C1_ESSA_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb, kernel_size=2, stride=2) # pool1
        x_rgb    = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2) # pool2
        x_rgb    = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        E_rgb_p3 = self.C3_ESSA_rgb(x_rgb_p3)
        x_rgb    = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2) # pool3
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2) # pool4
        x_rgb    = self.conv5_rgb(x_rgb)

        x_inf    = self.conv1_inf(x_inf)
        E_inf_p1 = self.C1_ESSA_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf, kernel_size=2, stride=2) # pool1
        x_inf    = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2) # pool2
        x_inf    = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        E_inf_p3 = self.C3_ESSA_inf(x_inf_p3)
        x_inf    = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2) # pool3
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2) # pool4
        x_inf    = self.conv5_inf(x_inf)

        x = torch.cat((x_rgb, x_inf), dim=1) # fusion RGB and INF

        # decode
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool4
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool3
        x_p3 = x + torch.cat((x_rgb_p3, x_inf_p3), dim=1)
        E_p3 = torch.cat([E_rgb_p3, E_inf_p3], dim=1)
        x = self.decode3(torch.cat([x_p3, E_p3], dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool2
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool1
        E_p1 = torch.cat([E_rgb_p1, E_inf_p1], dim=1)
        x = self.decode1(torch.cat([x, E_p1], dim=1))

        return x


if __name__ == '__main__':
    x = torch.rand([1, 4, 480, 640])
    model = MFNet_ESSA(n_class=9)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (1, 9, 480, 640), f'invalid shape: {y.shape}'

    print('[param_cnt]')
    print('  MFNet:',      sum(p.numel() for p in MFNet     (n_class=9).parameters()))
    print('  MFNet_ESSA:', sum(p.numel() for p in MFNet_ESSA(n_class=9).parameters()))

    print('Done!')
