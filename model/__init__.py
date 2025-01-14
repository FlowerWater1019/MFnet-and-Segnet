import torch.nn as nn

from .SegNet import SegNet
from .SegNet_ESSA import SegNet_ESSA
from .MFNet import MFNet
from .MFNet_ESSA import MFNet_ESSA
from .DeepLabV3 import DeepLabV3, get_dv3_model


MODELS = [
  'SegNet',
  'SegNet_ESSA',
  'MFNet',
  'MFNet_ESSA',
  'DeepLabV3',
]


def get_model(name:str, n_class:int, channels:str=None) -> nn.Module:
  if name in ['SegNet', 'SegNet_ESSA']:
    model = eval(name)(n_class=n_class, in_channels=channels)
  elif name in ['MFNet', 'MFNet_ESSA'] :
    model = eval(name)(n_class=n_class)
  elif name == 'DeepLabV3':
    model = get_dv3_model(n_class, channels)
  else:
    raise ValueError(f'unknow model name: {name}')
  return model
