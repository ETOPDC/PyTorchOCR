import pathlib
import numpy as np


def get_pathlist(data_path):
    """
    获取input和label
    :param datapaths: 记录文件,每行为一个样本，格式为 图片路径\t标签路径
    :return: 一个列表，[(img_path,label_path),...]
    """
    pathlist = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
            if len(line) > 1:
                # pathlib就是封装os.file的库
                img_path = pathlib.Path(line[0].strip(' '))
                label_path = pathlib.Path(line[1].strip(' '))
                # img_path.stat().st_size 返回文件大小
                if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                    pathlist.append((str(img_path), str(label_path)))
    return pathlist


def order_points_clockwise(pts):
    """
    对坐标进行顺时针排序
    :param pts: [[x,y],[],[],[]]
    :return:
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    # x+y最大和最小的
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    # x-y最大和最小的
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def create_transforms(op_param_list):
    from torchvision import transforms
    assert isinstance(op_param_list, list), "dataset-transform config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict), "dataset-transform yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        # getattr(transforms, xxx)(**param) 相当于 transforms.xxx(**args)
        op = getattr(transforms, op_name)(**param)
        ops.append(op)
    return transforms.Compose(ops)
