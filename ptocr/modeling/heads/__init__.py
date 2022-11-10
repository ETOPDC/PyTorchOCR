import copy
from .DBHead import DBHead
from .ConvHead import ConvHead

__all__ = ['build_head']
support_head = ['ConvHead', 'DBHead']


def build_head(config, in_channels):
    config = copy.deepcopy(config)
    head_name = config.pop("name")
    assert head_name in support_head, f'all support head is {support_head}'
    config["in_channels"] = in_channels
    head = eval(head_name)(**config)
    return head
