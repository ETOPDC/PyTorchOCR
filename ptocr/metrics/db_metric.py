import numpy as np

from .eval_det_iou import DetectionIoUEvaluator

# 用于求均值
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self
class DBMetric():
    def __init__(self, is_output_polygon=False):
        # 输出的是否是多边形或矩形
        self.is_output_polygon = is_output_polygon
        # 评估器，核心
        self.evaluator = DetectionIoUEvaluator(is_output_polygon=is_output_polygon)

    def measure(self, batch, output, box_thresh=0.6):
        """
        batch : 一个batch的gt图片信息
        output: 预测出的一个batch的文本框信息和概率均值，是一个tuple（batch文本框列表，batch概率均值列表）
        """

        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        # 传入的是一个batch，若干图片，每张图片有若干个文字框的gt
        gt_polyons_batch = batch['text_polys']
        ignore_tags_batch = batch['ignore_tags']
        # 0是boxes，1是score
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        # 得到这一张图片里的gt文字框，预测文字框，预测文字框概率均值，是否忽略，循环遍历每一张图片
        for polygons, pred_polygons, pred_scores, ignore_tags in zip(gt_polyons_batch, pred_polygons_batch,
                                                                     pred_scores_batch, ignore_tags_batch):
            # 将单个文字框的gt文字框和是否忽略整理成一个字典，放在list中
            gt = [dict(points=np.int64(polygons[i]), ignore=ignore_tags[i]) for i in range(len(polygons))]
            # 如果输出的是矩形，那么需要判断一下分数是否大于thresh，然后再整理
            if self.is_output_polygon:
                # 同上，存放整理后的预测文字框
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                pred = []
                # print(pred_polygons.shape)

                # 为什么输出是矩形的时候要加一步判断？一种可能的情况是输出是矩形，但是真实文字区域一个弯月牙，这样文字区域其实占这个矩形很小的部分，不利于后续识别，我们丢弃这样的区域
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(dict(points=pred_polygons[i, :, :].astype(np.int)))
                # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]

            # 核心功能，得到一张图片的p,r和hmean等信息
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, batch, output, box_thresh=0.6):
        return self.measure(batch, output, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    # 求整个数据集的最终结果
    # 把所有batch进行整合
    def gather_measure(self, raw_metrics):
        # raw_metrics应该是列表的列表的列表，两层for循环展开，单个元素代表一个图片的result
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]
        #
        result = self.evaluator.combine_results(raw_metrics)

        # 将数转换为对象
        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }