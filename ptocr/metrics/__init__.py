import copy
from .db_metric import DBMetric

__all__ = ['build_metric']
support_metric = ['DBMetric']


def build_metric(config):
    config = copy.deepcopy(config)
    metric_name = config.pop('name')
    assert metric_name in support_metric, f'all support loss is {support_metric}'
    metric = eval(metric_name)(config)
    return metric
