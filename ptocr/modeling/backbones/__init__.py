# from .表示从同级目录下导入
import copy
from .resnet import *
from .shufflenetv2 import *
from .MobilenetV3 import MobileNetV3

__all__ = ['build_backbone']

support_backbone = ['resnet18', 'deformable_resnet18', 'deformable_resnet50',
                    'resnet50', 'resnet34', 'resnet101', 'resnet152',
                    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                    'MobileNetV3']


def build_backbone(config):
    config = copy.deepcopy(config)
    backbone_name = config.pop("name")
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**config)
    return backbone
