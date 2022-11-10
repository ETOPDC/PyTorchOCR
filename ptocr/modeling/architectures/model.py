from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from addict import Dict
from torch import nn
import torch.nn.functional as F
from ptocr.modeling.backbones import build_backbone
from ptocr.modeling.necks import build_neck
from ptocr.modeling.heads import build_head


class DBNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # addict包下
        config = Dict(config)
        self.name = f'{config.algorithm}_{config.Backbone.name}_{config.Neck.name}_{config.Head.name}'
        self.backbone = build_backbone(config.Backbone)
        self.neck = build_neck(config.Neck, in_channels=self.backbone.out_channels)
        self.head = build_head(config.Head, in_channels=self.neck.out_channels)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        # 这一步可能是为了保险，没啥必要
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    import torch

    device = torch.device('cpu')
    x = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'Backbone': {'name': 'resnet18', 'pretrained': True, "in_channels": 3},
        'Neck': {'name': 'DBFPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'Head': {'name': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model_config = Dict(model_config)
    model = DBNet(config=model_config).to(device)
    import time

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    print(model.name)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
