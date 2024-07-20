import os
import torch
import sys
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import random
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale


# run Dif-Fusion for the MF dataset
# modified from Dif-Fusion/t_fusion.py

parser = argparse.ArgumentParser()
parser.add_argument('-I', '--in_dp', type=Path, default='data/MF', help='input image path')
parser.add_argument('-O', '--out_dp', type=Path, default='data/MF_Dif-Fusion', help='output image path')
parser.add_argument('-c', '--config', type=str, default='config/fusion_test.json',
                        help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='test')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-log_eval', action='store_true')
args = parser.parse_args()

# NOTE: first expand paths to absolute here!!
args.in_dp = Path(args.in_dp).absolute()
args.out_dp = Path(args.out_dp).absolute()
print('>> in_dp:', args.in_dp)
print('>> out_dp:', args.out_dp)
assert args.in_dp.is_dir()

# NOTE: then we can safely chdir to repo root path!!
try:
  os.chdir(os.path.join('repo', 'Dif-Fusion'))
  print('>> cwd:', os.getcwd())
  sys.path.append(os.getcwd())
except:
  print('>> Error: run repo\init_repos.cmd first')
  exit(0)

import models as Model
import core.logger as Logger
import core.metrics as Metrics


def get_img(content, name):
  img_org = Image.open(os.path.join(content, name))
  assert img_org.mode == 'RGBA' or img_org.mode == 'RGBa'
  r, g, b, ir = img_org.split()
  img_rgb = Image.merge('RGB', (r, g, b))
  img_ir = ir.convert('L')
  
  min_max = (-1, 1)
  img_rgb = ToTensor()(img_rgb)
  img_rgb = img_rgb * (min_max[1] - min_max[0]) + min_max[0]
  
  img_ir = ToTensor()(img_ir)
  img_ir = img_ir * (min_max[1] - min_max[0]) + min_max[0]
  
  img_cat = torch.cat([img_rgb, img_ir[0:1, :, :]], axis=0).unsqueeze(0)
  img_rgb, img_ir = img_rgb.unsqueeze(0), img_ir.unsqueeze(0)
  
  return {'img': img_cat, 'vis': img_rgb, 'ir': img_ir[0:1, :, :]}
  
  
def infer(args):
  opt = Logger.parse(args)
  opt = Logger.dict_to_nonedict(opt)
  diffusion = Model.create_model(opt)
  fussion_net = Model.create_fusion_model(opt)

  img_dp = os.path.join(args.in_dp, 'images')
  for img_name in tqdm(os.listdir(img_dp)):
    test_data = get_img(img_dp, img_name)
    diffusion.feed_data(test_data)
    fes, fds = [], []
  
    for t in opt['model_df']['t']:
      fe_t, fd_t = diffusion.get_feats(t=t)
      if opt['model_df']['feat_type'] == "dec":
        fds.append(fd_t)
        del fd_t
      else:
        fes.append(fe_t)
        del fe_t
    
    fussion_net.feed_data(fds, test_data)
    fussion_net.test()
    visuals = fussion_net.get_current_visuals()
    grid_img = visuals['pred_rgb'].detach()
    grid_img = Metrics.tensor2img(grid_img)
    Metrics.save_img(grid_img, os.path.join(args.out_dp, img_name))


if __name__ == '__main__':
  print('[infer]')
  infer(args)
  