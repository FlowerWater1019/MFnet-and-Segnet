import torch.nn as nn

from .MFNet import MFNet
from .SegNet import SegNet
from .DeepLabV3 import DeepLabV3, get_dv3_model


def get_model(name:str, n_class:int, channels:str=None) -> nn.Module:
  if name == 'SegNet':
    model = eval(name)(n_class=n_class, in_channels=channels)
  elif name == 'MFNet':
    model = eval(name)(n_class=n_class)
  elif name == 'DeepLabV3':
    model = get_dv3_model(n_class, channels)
  else:
    raise ValueError(f'unknow model name: {name}')

  return model
