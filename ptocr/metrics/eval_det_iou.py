from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
import cv2

# 计算iou值
def iou_rotate(box_a, box_b, method='union'):
    rect_a = cv2.minAreaRect(box_a)
    rect_b = cv2.minAreaRect(box_b)
    # 返回重叠部分 点的坐标
    r1 = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if r1[0] == 0:
        return 0
    else:
        inter_area = cv2.contourArea(r1[1])
        area_a = cv2.contourArea(box_a)
        area_b = cv2.contourArea(box_b)
        union_area = area_a + area_b - inter_area
        if union_area == 0 or inter_area == 0:
            return 0
        if method == 'union':
            iou = inter_area / union_area
        elif method == 'intersection':
            iou = inter_area / min(area_a, area_b)
        else:
            raise NotImplementedError
        return iou


# 标准检测iou评价函数
class DetectionIoUEvaluator(object):
    def __init__(self, is_output_polygon=False, iou_constraint=0.5, area_precision_constraint=0.5):
        """
        iou阈值
        """
        self.is_output_polygon = is_output_polygon
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    # 评价核心函数，对iou进行评价
    def evaluate_image(self, gt, pred):
        """
        输入
        gt：list[dict{}]，一张图片里所有的gt文字框，单个字典是一个gt文字框的点坐标和是否忽略
        pred：list[dict{}]，单个字典是一个预测文字框
        """

        # 求并集
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        # 求iou
        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        # 求交集
        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        # pred和gt匹配的总个数
        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        # 整张图片里care的gt
        numGlobalCareGt = 0
        # 整张图片里care的预测文字框
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        # 预测文本框匹配的个数
        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        # 不care的gt框的索引列表
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        # 不care的预测框的索引
        detDontCarePolsNum = []

        # pred和gt的匹配对
        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        # 将一张图片的gt文字框和是否忽略加到gtPols和gtPolPoints
        for n in range(len(gt)):
            # 获取一个文字框的点坐标
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']

            #
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)  # 这一步有点多余，下面没用到
            if dontCare:
                # 记录下第几个文字框需要被忽略，gtDontCarePolsNum里存储的是gtPols的索引
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(
            gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        # 遍历每个预测框
        for n in range(len(pred)):
            # 获取每个框的点坐标
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)

            # 下面这一块的意思是，如果当前这个预测框，与某个需要忽略的gt重叠部分超过阈值，那么这个预测框也要被忽略
            # 如果有需要忽略的gt文字框
            if len(gtDontCarePolsNum) > 0:
                # 遍历这些需要忽略的gt文字框
                for dontCarePol in gtDontCarePolsNum:
                    # 获取需要忽略的gt文字框点坐标
                    dontCarePol = gtPols[dontCarePol]
                    # 计算当前这个预测框和这个需要忽略的gt框的交集的面积
                    intersected_area = get_intersection(dontCarePol, detPol)
                    # 求预测框的面积，
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    # 如果当前这个预测框，与某个需要忽略的gt重叠部分超过阈值，也就是太大，这个预测框也要被忽略
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(
            detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")

        # ***************************************************************重点
        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            # 搞了个iou混淆矩阵
            iouMat = np.empty(outputShape)

            # vis数组，深度优先搜索方法
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            # 如果输出框是多边形，直接暴力计算相应的iou，填充iouMat
            if self.is_output_polygon:
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
            # 如果输出是矩形
            else:
                # gtPols = np.float32(gtPols)
                # detPols = np.float32(detPols)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = np.float32(gtPols[gtNum])
                        pD = np.float32(detPols[detNum])
                        iouMat[gtNum, detNum] = iou_rotate(pD, pG)

            # 计算好iouMat后
            # 继续暴力循环
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    # 如果当前gtNum的框和detNum的框都还没有被配对，并且这两个框都不是要忽略的
                    if gtRectMat[gtNum] == 0 and detRectMat[
                        detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        # 通过iou阈值寻找gt和pred匹配的文字框
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1  # 标记上
                            detMatched += 1  # pred框匹配个数加1
                            pairs.append({'gt': gtNum, 'det': detNum})  # pair加上新的一对
                            detMatchedNums.append(detNum)  # 保存匹配好的pred框的索引
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

        # gt个数n pred m  match匹配个数 k
        # recall =  k/n   precsion = k/m
        # 计算真正保留下的gt和pred的文字框个数
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            # 匹配上的个数/ 有效的gt个数
            recall = float(detMatched) / numGtCare
            # 匹配上的个数/ 有效的检测个数
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare

        # f1score 调和评价   阿尔法取1
        hmean = 0 if (precision + recall) == 0 else 2.0 * \
                                                    precision * recall / (precision + recall)

        # 没有用
        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        # 这是原始别人的代码，作者进行了重写
        # matchedSum += detMatched
        # numGlobalCareGt += numGtCare
        # numGlobalCareDet += numDetCare
        #
        # perSampleMetrics = {
        #     'gtCare': numGtCare,
        #     'detCare': numDetCare,
        #     'detMatched': detMatched,
        # }
        # return perSampleMetrics


        # 这里传回的只是一张图片的，返回一大串，好像没啥用
        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            ####
            'gtCare': numGtCare,
            'detCare': numDetCare,
            ####
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            ###
            'detMatched': detMatched,
            ####
            'evaluationLog': evaluationLog
        }

        return perSampleMetrics

    # 计算整个dataloader所有batch的平均结果
    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare'] # recall分母
            numGlobalCareDet += result['detCare'] # precision分母
            matchedSum += result['detMatched'] # 分子

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)

        methodMetrics = {'precision': methodPrecision,
                         'recall': methodRecall, 'hmean': methodHmean}

        return methodMetrics


if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    preds = [[{
        'points': [(0.1, 0.1), (0.5, 0), (0.5, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(0.5, 0.1), (1, 0), (1, 1), (0.5, 1)],
        'text': 5678,
        'ignore': False,
    }]]
    gts = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)