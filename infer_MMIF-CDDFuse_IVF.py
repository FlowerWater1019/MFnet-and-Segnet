import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import logging ; logging.basicConfig(level=logging.CRITICAL)
import warnings ; warnings.filterwarnings("ignore") 

import cv2
import torch
from tqdm import tqdm
import numpy as np

# run MMIF-CDDFuse Infrared-Visible Fusion for the MF dataset
# modified from MMIF-CDDFuse/test_IVF.py

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = r"models/CDDFuse_IVF.pth"

parser = ArgumentParser()
parser.add_argument('-I', '--in_dp', type=Path, default='data/MF', help='input image path')
parser.add_argument('-O', '--out_dp', type=Path, default='data/MF_CDDFuse_IVF', help='output image path')
args = parser.parse_args()

# NOTE: first expand paths to absolute here!!
args.in_dp = Path(args.in_dp).absolute()
args.out_dp = Path(args.out_dp).absolute()
print('>> in_dp:', args.in_dp)
print('>> out_dp:', args.out_dp)
assert args.in_dp.is_dir()

# NOTE: then we can safely chdir to repo root path!!
try:
  os.chdir(os.path.join('repo', 'MMIF-CDDFuse'))
  print('>> cwd:', os.getcwd())
  sys.path.append(os.getcwd())
except:
  print('>> Error: run repo\init_repos.cmd first')
  exit(0)
from utils.img_read_save import img_save
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction


def image_read_cv2_bundle(path):
  img_BGRA = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
  img_BGR = img_BGRA[..., :3]
  img_IF = img_BGRA[..., 3]
  img_RGB = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
  return img_RGB, img_IF


@torch.inference_mode()
def infer(args):
  Encoder = torch.nn.DataParallel(Restormer_Encoder()).to(device)
  Decoder = torch.nn.DataParallel(Restormer_Decoder()).to(device)
  BaseFuseLayer = torch.nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
  DetailFuseLayer = torch.nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
  Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
  Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
  BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
  DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
  Encoder.eval()
  Decoder.eval()
  BaseFuseLayer.eval()
  DetailFuseLayer.eval()

  img_dp = os.path.join(args.in_dp, "images")
  for img_name in tqdm(os.listdir(img_dp)):
    img_RGB, img_IF = image_read_cv2_bundle(os.path.join(img_dp, img_name))
    img_RGB, img_IF = img_RGB/255.0, img_IF/255.0
    data_IR = img_IF[np.newaxis,np.newaxis, ...]
    data_VIS = img_RGB[np.newaxis,np.newaxis, ...]

    data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
    data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

    feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
    feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
    feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
    feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
    data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
    data_Fuse = (data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
    fi = (data_Fuse * 255).byte().cpu().numpy().squeeze()
    img_save(fi, img_name.split(sep='.')[0], args.out_dp)


if __name__ == '__main__':
  print('[infer]')
  infer(args)
