import json
import pathlib

import cv2
import numpy as np
from matplotlib import pyplot as plt


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    # pathlib就是封装os.file的库
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    # img_path.stat().st_size 返回文件大小
                    if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)

def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content

def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content

def expand_polygon(polygon):
    """
    对只有一个字符的框进行扩充
    """
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


def parse_config(config: dict) -> dict:
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config

def show_img(imgs: np.ndarray, title='img'):
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()

def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path
