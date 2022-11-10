import argparse


def get_options(parser=argparse.ArgumentParser(description="PyTorchOCR")):
    # parser.add_argument('--config_file', default='config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--config_file', default='configs/icdar2015_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
    opt = parser.parse_args()
    return opt
