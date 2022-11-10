import copy
from .model import DBNet

__all__ = ['build_model']
support_model = ['DBNet']


def build_model(config):
    """
    get architecture model class
    """
    config = copy.deepcopy(config)
    arch_type = config.pop('algorithm')
    assert arch_type in support_model, f'{arch_type} is not developed yet!, only {support_model} are support now'
    arch_model = eval(arch_type)(config)
    return arch_model
