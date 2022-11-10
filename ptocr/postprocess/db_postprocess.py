import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


# db后处理，将模型输出的概率图转换为文本框列表
class DBPostProcess(object):
    """
    db后处理，将模型输出的概率图转换为文本框列表

    输入：
    batch：一个batch的图片信息，是一个字典，key是string，value是list，list里是每张图片的信息，
          具体传入属性由自定义dataset决定，重要的属性有，text_polys gt文本框，ignore_tags 是否可忽略，
          img 图片信息三通道[N,C,H,W]，shape 图片原始形状，
    pred：模型输出一个batch的结果，shape为[N,C,H,W]，单张图片，通道0是概率图，1是阈值图，2是二值化图
    is_output_polygon：是输出多边形，还是输出四边矩形
    输出：
    boxes_batch：一个batch图片的文字框，是一个list，单个元素是一个图片的所有预测的文字区域
    scores_batch：一个batch图片的文字框平均概率，是一个list，单个元素是一个图片的预测的文字区域的平均概率，由概率图求得
    """

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        """
        thresh：后处理使用的传统二值化的阈值
        box_thresh：单个文字区域置信度的阈值，小于则丢弃这个文字区域
        max_candidates：单张图片文字区域的最大个数
        unclip_ratio：膨胀公式的膨胀因子 r'
        """
        # 文字区域最小边长，小于3丢弃
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    # 通过二值图来寻找文字多边形框，通过概率图求文字区域的概率均值
    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        通过二值图来寻找文字多边形框，通过概率图求文字区域的概率均值

        输入
        pred：概率图
        _bitmap：通过传统二值化后得到的二值化图
        dest_width：原始图片宽
        dest_height：原始图片高

        输出
        boxes：单张图片的预测文本框列表
        scores：预测文本框的概率均值
        """

        # 下一步去寻找轮廓
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        # 在预测步骤resize之后的高和宽
        height, width = bitmap.shape
        boxes = []
        scores = []

        # contours是一个列表，一系列点的坐标，表示了整个图里面所有的轮廓，即所有的文本框轮廓
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # contour是一个列表，有若干个点坐标
        for contour in contours[:self.max_candidates]:  # 遍历每个文本框轮廓
            # arcLength求轮廓周长
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)  # 这两步的作用是将原来相对光滑曲线（点太多）转换为折线，True表示仍然是封闭的

            points = approx.reshape((-1, 2))  # 获取点坐标
            if points.shape[0] < 4:  # 至少是四边形，否则跳过
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            # 另一步筛选，传入参数是（概率图，一个文字区域若干个点坐标），得到一个文字区域概率的均值
            score = self.box_score_fast(pred, contour.squeeze(1))
            # 如果求出来的文字区域概率平均值小于概率阈值，丢弃
            # 置信度不够高，不要
            if self.box_thresh > score:
                continue

            # 至少要三个点，才是个封闭区域，才能执行膨胀操作
            if points.shape[0] > 2:
                # 膨胀操作，返回扩张之后点的坐标，点的个数不变
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                # 判断是否出现两个以上区域（膨胀不会出现，不可能大于一，shrink可能出现，一个大区域分裂成两个）
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            # 传入参数 [[1,1],[2,3]] -> [[[1,1]],[[2,2]]] 膨胀后点坐标
            # 得到最小边长
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            # 保证最小边长不能太小，即文字框不能太小，太小就不要了
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            # 因为预测阶段（第83行）进行了resize，这里重新resize到原图尺度下，点的坐标
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    # 通过二值图寻找文字矩形(四个点)
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        # 遍历每一个轮廓
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            # 得到顺时针排列的四个点坐标，以及最小边长
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            # 得到这个最小矩形的文本概率均值（得分）
            score = self.box_score_fast(pred, contour)
            # 置信度不够，丢弃
            if self.box_thresh > score:
                continue

            # 膨胀
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)

            # 对膨胀后，重复
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            # 得到原图尺度下的边框坐标
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    # 固定套路算法，对文字框进行膨胀或收缩
    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    # 获取能够包含这些点的最小矩形
    def get_mini_boxes(self, contour):
        # 返回能够包含这些点的最小矩形（包含该矩形的中心点坐标、高度宽度及倾斜角度等信息）
        bounding_box = cv2.minAreaRect(contour)
        # 使用cv2.boxPoints()可获取该矩形的四个顶点坐标，然后按照x坐标进行排序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # 将四个点从左上角开始，按照顺时针排列
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    # 计算文字区域概率的均值
    def box_score_fast(self, bitmap, _box):
        """
        计算文字区域概率的均值

        bitmap：概率图
        _box：一个文字区域若干个点坐标，shape为[X,2]
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        # 移动坐标系
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        # 通过cv2.mean计算单个文字区域概率均值，通过cv2.fillPoly做一个掩膜mask，所以刚才需要移动坐标系，多边形的坐标从0开始，如果不从0开始，会有偏移
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        # 计算文字区域概率的均值
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, batch, pred, is_output_polygon=False):
        # 获取概率图shrink map
        pred = pred[:, 0, :, :]
        # 传统二值化，阈值为0.3
        segmentation = pred > self.thresh
        # 存放batch图片的所有文本框坐标
        boxes_batch = []
        # 存放batch图片的所有文本框平均置信度
        scores_batch = []
        # 遍历batch，inference预测阶段只有一个
        for batch_index in range(pred.size(0)):
            # 原始图片的高度和宽度
            height, width = batch['shape'][batch_index]
            # 是否输出多边形的框
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch
