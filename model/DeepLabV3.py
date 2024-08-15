# coding:utf-8
import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.segmentation import DeepLabV3, deeplabv3_resnet50


def get_dv3_model(n_class:int, in_channels:int) -> DeepLabV3:
    model: DeepLabV3 = deeplabv3_resnet50(pretrained=False, num_classes=n_class)
    # hijack the first layer
    backbone: ResNet = model.backbone
    conv1_old = backbone.conv1
    conv1_new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv1_old.out_channels,
        kernel_size=conv1_old.kernel_size,
        stride=conv1_old.stride,
        padding=conv1_old.padding,
        bias=conv1_old.bias is not None
    )
    backbone.conv1 = conv1_new
    # fix .forward() output
    model.org_forward = model.forward
    model.forward = lambda x: model.org_forward(x)['out']
    return model
