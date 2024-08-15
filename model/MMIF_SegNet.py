import logging ; logging.basicConfig(level=logging.CRITICAL)
import warnings ; warnings.filterwarnings("ignore") 

import os
import sys
from PIL import Image

import torch
import torch.nn as nn
import numpy as np

from SegNet import SegNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = r"models/CDDFuse_IVF.pth"
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
os.chdir(os.path.join(root_dir, 'repo', 'MMIF-CDDFuse'))
print('>> cwd:', os.getcwd())
sys.path.append(os.getcwd())

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction


class MMIF_SegNet(nn.Module):

    def __init__(self, n_class, in_channels):
        super(MMIF_SegNet, self).__init__()

        self.in_channels = in_channels
        self.Encoder = torch.nn.DataParallel(Restormer_Encoder()).to(device)
        self.Decoder = torch.nn.DataParallel(Restormer_Decoder()).to(device)
        self.BaseFuseLayer = torch.nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
        self.DetailFuseLayer = torch.nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
        self.SegNet = SegNet(n_class, in_channels)

    def forward(self, data_vis, data_ir):
        feature_V_B, feature_V_D, feature_V = self.Encoder(data_vis)
        feature_I_B, feature_I_D, feature_I = self.Encoder(data_ir)
        feature_F_B = self.BaseFuseLayer(feature_V_B, feature_I_B)
        feature_F_D = self.DetailFuseLayer(feature_V_D, feature_I_D)
        data_Fuse, _ = self.Decoder(data_vis, feature_F_B, feature_F_D)
        data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
        fi = (data_Fuse * 255).byte().cpu().numpy().squeeze()
        
        input_w, input_h = 640, 480
        img = np.asarray(Image.fromarray(fi).resize((input_w, input_h)), dtype=np.float32)
        img = np.expand_dims(img, axis=0) / 255
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        pred_img = self.SegNet(img)
        return pred_img
    
    
def unit_test():
    model = MMIF_SegNet(n_class=9, in_channels=1)
    data_vis, data_ir = torch.rand(1, 1, 480, 640).to(device), torch.rand(1, 1, 480, 640).to(device)
    y = model(data_vis, data_ir)
    print(y.shape)
    assert y.shape == torch.Size([1, 9, 480, 640])
    print('Unit test pass')
    
    
if __name__ == '__main__':
    unit_test()
