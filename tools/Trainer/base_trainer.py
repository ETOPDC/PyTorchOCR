import copy
import os.path
import pathlib
import shutil

import anyconfig
import torch

from .util import init_logger
from pprint import pformat


class BaseTrainer():
    def __int__(self, config, model, criterion, optimizer, lr_scheduler):
        config = copy.deepcopy(config)
        config["Global"]["output_dir"] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent),
                                                      config["Global"]["output_dir"])
        self.save_dir = os.path.join(config["Global"]["output_dir"], model.name)
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoint")

        # 不配置就删除，慎用
        if config['trainer']['resume_checkpoint'] == '' and config['trainer']['finetune_checkpoint'] == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = 0
        self.start_epoch = 0
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # logger and tensorboard
        self.tensorboard_enable = self.config["Global"]["tensorboard"]
        self.epochs = self.config["Global"]["epoch_num"]
        self.log_iter = self.config["Global"]["log_iter"]
        if config["local_rank"] == 0:
            anyconfig.dump(config, os.path.join(self.save_dir, "config.yaml"))
            self.logger = init_logger(os.path.join(self.save_dir, "train.log"))
            self.logger_info(pformat(self.config))

        # device
        torch.manual_seed(self.config["Global"]["seed"])  # 为CPU设置随机种子
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.config["Global"]["seed"])  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config["Global"]["seed"])  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.device = torch.device("cpu")
        self.logger_info(f"Using device {self.device} and pytorch {torch.__version__}")

        # metrics
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}

        # resume or finetune
        if self.config['trainer']['resume_checkpoint'] == '':
            self.logger_info("Loading checkpoint: {} ...".format(self.config['trainer']['resume_checkpoint']))
            checkpoint = torch.load(config['trainer']['resume_checkpoint'], map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True) # 加载模型可学习参数
            if "metrics" in







    def logger_info(self, s):
        if self.config['local_rank'] == 0:
            self.logger.info(s)

    def _load_checkpoint(self, , resume):

