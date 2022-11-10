import copy
from .DBFPN import DBFPN
from .FPEM_FFM import FPEM_FFM

__all__ = ['build_neck']
support_neck = ['DBFPN', 'FPEM_FFM']


def build_neck(config, in_channels):
    config = copy.deepcopy(config)
    neck_name = config.pop("name")
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    config["in_channels"] = in_channels
    neck = eval(neck_name)(**config)
    return neck
