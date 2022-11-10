import copy
from torch.utils.data import DataLoader
from .utils import create_transforms
# import dataset
from . import dataset

__all__ = ["build_dataloader"]


def build_dataloader(config, distributed=False):
    if config is None:
        return None
    config = copy.deepcopy(config)
    dataset_config = config.pop("dataset")
    loader_config = config.pop("loader")

    # 创建transform
    if "transforms" in dataset_config:
        transform_opt = create_transforms(dataset_config.pop("transforms"))
    else:
        transform_opt = None

    # 创建dataset
    _dataset = getattr(dataset, dataset_config["name"])(data_path=dataset_config.pop("data_path"),
                                                        img_mode=dataset_config.pop("img_mode"),
                                                        pre_processes=dataset_config.pop("pre_processes"),
                                                        keep_keys=dataset_config.pop("keep_keys"),
                                                        filter_keys=dataset_config.pop("filter_keys"),
                                                        ignore_tags=dataset_config.pop("ignore_tags"),
                                                        transform=transform_opt)

    # 配置collect_fn
    if "collate_fn" not in loader_config or loader_config["collate_fn"] is None or len(
            loader_config["collate_fn"]) == 0:
        loader_config["collate_fn"] = None
    else:
        loader_config["collate_fn"] = eval(loader_config["collate_fn"])()

    # 创建dataloader
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(_dataset)
        loader_config["shuffle"] = False
        loader_config["pin_memory"] = True
    loader = DataLoader(dataset=_dataset, sampler=sampler, **loader_config)
    return loader
