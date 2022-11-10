import os
import random
import numpy as np
import torch
from tqdm import tqdm
from ptocr import *


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),
                                             rank=opt.local_rank)
        config["distributed"] = True
    else:
        config["distributed"] = False
    config["local_rank"] = opt.local_rank

    # dataloader
    train_dataloader = build_dataloader(config["Train"], distributed=config["distributed"])
    assert train_dataloader is not None, "train loader is None."
    if "Validate" in config:
        validata_dataloader = build_dataloader(config["Validate"], distributed=False)
    else:
        validata_dataloader = None

    # criterion
    criterion = build_loss(config['Loss']).to(device)
    config["Architecture"]["Backbone"]["in_channels"] = 3 if config["Train"]["dataset"]["img_mode"] != "GRAY" else 1

    # model
    model = build_model(config["Architecture"]).to(device)

    # 函数，将概率图转换为文本框
    post_p = build_post_processing(config["PostProcess"])

    # 评估
    metric = build_metric(config["Metric"])

    optimizer, lr_scheduler = build_optimizer()



    Trainer(config=config)



# def parse_config(config: dict) -> dict:
#     import anyconfig
#     base_file_list = config.pop('base')
#     base_config = {}
#     for base_file in base_file_list:
#         tmp_config = anyconfig.load(open(base_file, 'rb'))
#         if 'base' in tmp_config:
#             tmp_config = parse_config(tmp_config)
#         anyconfig.merge(tmp_config, base_config)
#         base_config = tmp_config
#     anyconfig.merge(base_config, config)
#     return base_config


# 随机数的设置，保证复现结果
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import sys
    import pathlib

    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    project = 'PyTorchOCR'  # 工作项目根目录
    sys.path.append(os.getcwd().split(project)[0] + project)
    print(sys.path)
    os.chdir("../")

    import config
    import anyconfig

    opt = config.get_options()
    assert os.path.exists(opt.config_file), f"{opt.config_file} file not exist."
    config_from_file = anyconfig.load(open(opt.config_file, 'rb'))

    manual_seed = config_from_file["Global"]["seed"]
    set_seed(manual_seed)

    import anyconfig

    main(config_from_file)
