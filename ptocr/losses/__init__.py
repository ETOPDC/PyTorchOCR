import copy
from .db_loss import DBLoss

__all__ = ['build_loss']
support_loss = ['DBLoss']

def build_loss(config):
    config = copy.deepcopy(config)
    loss_name = config.pop('name')
    assert loss_name in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_name)(config)
    return criterion