import copy
from .lr_scheduler import WarmupPolyLR

__all__ = ['build_optimizer']
support_optimizer = ['Adam']
support_lr_scheduler = ["WarmupPolyLR"]


def build_optimizer(config, epochs, step_each_epoch, model):
    import torch.optim
    config = copy.deepcopy(config)
    optimizer_name = config.pop("name")
    lr_scheduler_config = config.pop("Lr_scheduler")
    lr_scheduler_name = lr_scheduler_config.pop("name")
    assert optimizer_name in support_optimizer, f'all support optimizer is {support_optimizer}'
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **config)
    assert lr_scheduler_name in support_lr_scheduler, f'all support lr scheduler is {support_lr_scheduler}'
    lr_scheduler = eval(lr_scheduler_name)(optimizer=optimizer, max_iters=epochs * step_each_epoch,
                                           warmup_iters=lr_scheduler_config["warmup_epoch"] * step_each_epoch,
                                           **lr_scheduler_config)

    return optimizer, lr_scheduler
