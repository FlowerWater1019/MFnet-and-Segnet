import os
import sys
import argparse
from PIL import Image

import torch
import torch.nn as nn
import numpy as np

from SegNet import SegNet


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
os.chdir(os.path.join(root_dir, 'repo', 'Dif-Fusion'))
print('>> cwd:', os.getcwd())
sys.path.append(os.getcwd())
  
  
import models as Model
import core.logger as Logger
import core.metrics as Metrics


args = argparse.Namespace(config='config/df_segnet_train.json', 
                          debug=False, enable_wandb=False, 
                          log_eval=False, phase='train',
                          gpu_ids=None)


class DF_SegNet(nn.Module):

    def __init__(self, n_class, in_channels):
        super(DF_SegNet, self).__init__()

        self.in_channels = in_channels
        self.opt = Logger.parse(args)
        self.opt = Logger.dict_to_nonedict(self.opt)

        self.SegNet = SegNet(n_class, in_channels)
        self.diffusion = Model.create_model(self.opt)
        self.fussion_net = Model.create_fusion_model(self.opt)

    def forward(self, x):
        self.diffusion.feed_data(x)
        fes, fds = [], []

        for t in self.opt['model_df']['t']:
            fe_t, fd_t = self.diffusion.get_feats_grad(t=t)
                
            if self.opt['model_df']['feat_type'] == "dec":
                fds.append(fd_t)
                del fd_t
            else:
                fes.append(fe_t)
                del fe_t
    
        self.fussion_net.feed_data(fds, x)
        self.fussion_net.test_grad()
        visuals = self.fussion_net.get_current_visuals()
        grid_img = visuals['pred_rgb'].detach()
        grid_img = Metrics.tensor2img(grid_img)

        input_w, input_h = 640, 480
        grid_img = np.asarray(Image.fromarray(grid_img).resize((input_w, input_h)), dtype=np.float32).transpose((2,0,1))/255
        grid_img = torch.from_numpy(grid_img).unsqueeze(0)

        pred_img = self.SegNet(grid_img)
        return pred_img
        
        
def unit_test():
    x = {
        'img': torch.randn(1, 4, 480, 640), 
        'vis': torch.randn(1, 3, 480, 640), 
        'ir':  torch.randn(1, 1, 480, 640),
    }
    model = DF_SegNet(n_class=9, in_channels=3)
    y = model(x)
    print(y.shape)
    assert y.shape == torch.Size([1, 9, 480, 640])
    print('Unit test pass')
    
    
if __name__ == '__main__':
    unit_test()
