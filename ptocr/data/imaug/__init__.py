# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 10:53
# @Author  : zhoujun
from .iaa_augment import IaaAugment
from .augment import *
from .random_crop_data import EastRandomCropData, PSERandomCrop
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap

__all__ = ["create_operators"]


def create_operators(op_param_list, global_config=None):
    assert isinstance(op_param_list, list), ('pre processes operator config should be a list')
    ops = []
    for operator in op_param_list:
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param) if isinstance(param, dict) else eval(op_name)(param)
        ops.append(op)
    return ops
