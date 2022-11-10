import pathlib
import os
import cv2
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

from .imaug.utils import load, expand_polygon

from .utils import get_pathlist, order_points_clockwise
# from ptocr.data.base_dataset import BaseDataSet
from .base_dataset import BaseDataSet


class ICDAR2015Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, keep_keys, filter_keys, ignore_tags, transform=None):
        super().__init__(data_path, img_mode, pre_processes, keep_keys, filter_keys, ignore_tags, transform)

    def load_data(self, data_path) -> list:
        # 返回一个list,[(str(img_path), str(label_path)),...]
        path_list = get_pathlist(data_path)

        data_list = []
        for img_path, label_path in path_list:
            annotation = self._get_annotation(label_path)
            if len(annotation['text_polys']) > 0:
                data = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                data.update(annotation)
                data_list.append(data)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    # 点顺时针排序
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    # 计算轮廓的面积
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except:
                    print('load label failed on {}'.format(label_path))
        annotation = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return annotation


class DetDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        self.load_char_annotation = kwargs['load_char_annotation']
        self.expand_one_char = kwargs['expand_one_char']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        从json文件中读取出 文本行的坐标和gt，字符的坐标和gt
        :param data_path:
        :return:
        """
        data_list = []
        for path in data_path:
            content = load(path)
            for gt in tqdm(content['data_list'], desc='read file {}'.format(path)):
                img_path = os.path.join(content['data_root'], gt['img_name'])
                polygons = []
                texts = []
                illegibility_list = []
                language_list = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0 or len(annotation['text']) == 0:
                        continue
                    if len(annotation['text']) > 1 and self.expand_one_char:
                        annotation['polygon'] = expand_polygon(annotation['polygon'])
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    illegibility_list.append(annotation['illegibility'])
                    language_list.append(annotation['language'])
                    if self.load_char_annotation:
                        for char_annotation in annotation['chars']:
                            if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                                continue
                            polygons.append(char_annotation['polygon'])
                            texts.append(char_annotation['char'])
                            illegibility_list.append(char_annotation['illegibility'])
                            language_list.append(char_annotation['language'])
                data_list.append({'img_path': img_path, 'img_name': gt['img_name'], 'text_polys': np.array(polygons),
                                  'texts': texts, 'ignore_tags': illegibility_list})
        return data_list


class SynthTextDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, transform=None, **kwargs):
        self.transform = transform
        self.dataRoot = pathlib.Path(data_path)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, transform)

    def load_data(self, data_path: str) -> list:
        t_data_list = []
        for imageName, wordBBoxes, texts in zip(self.imageNames, self.wordBBoxes, self.transcripts):
            item = {}
            wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
            _, _, numOfWords = wordBBoxes.shape
            text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
            text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
            transcripts = [word for line in texts for word in line.split()]
            if numOfWords != len(transcripts):
                continue
            item['img_path'] = str(self.dataRoot / imageName)
            item['img_name'] = (self.dataRoot / imageName).stem
            item['text_polys'] = text_polys
            item['texts'] = transcripts
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]
            t_data_list.append(item)
        return t_data_list


if __name__ == '__main__':
    import torch
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from imaug.utils import show_img, plt, draw_bbox

    os.chdir("../..")
    print(os.getcwd())
    config = anyconfig.load('configs/icdar2015_resnet18_FPN_DBhead_polyLR.yaml')
    deep_config = config["Train"]
    dataset_config = deep_config.pop("dataset")
    loader_config = deep_config.pop("loader")

    train_dataset = ICDAR2015Dataset(data_path=dataset_config.pop("data_path"),
                                     transform=transforms.ToTensor(),
                                     img_mode=dataset_config.pop("img_mode"),
                                     pre_processes=dataset_config.pop("pre_processes"),
                                     keep_keys=dataset_config.pop("keep_keys"),
                                     filter_keys=dataset_config.pop("filter_keys"),
                                     ignore_tags=dataset_config.pop("ignore_tags"))

    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, data in enumerate(tqdm(train_loader)):
        # img = data['img']
        # shrink_label = data['shrink_map']
        # threshold_label = data['threshold_map']
        #
        # print(threshold_label.shape, threshold_label.shape, img.shape)
        # show_img(img[0].numpy().transpose(1, 2, 0), title='image')
        # show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label')
        # show_img((threshold_label[0].to(torch.float)).numpy(), title='threshold_label')
        # img = draw_bbox(img[0].numpy().transpose(1, 2, 0), np.array(data['text_polys']))
        # show_img(img, title='draw_bbox')
        # plt.show()
        pass
