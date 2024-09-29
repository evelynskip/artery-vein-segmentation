
"""https://blog.csdn.net/sinat_29047129/article/details/103642140?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control"""
"""https://blog.csdn.net/m0_47355331/article/details/119972157""" 

import torch
import cv2
import numpy as np
__all__ = ['SegmentationMetric']

"""
confusionMetric  
L\P     P    N
P      TP    FP
N      FN    TN
"""


import numpy as np
import torch
class SegmentationMetric(object):
    def __init__(self, numClass,ignore_labels=[]):
        self.ignore_labels = ignore_labels
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # ignore specified labels in imgLabel
        # for IgLabel in self.ignore_labels:
        #     mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = np.nanmean(IoU[1:]) # 求各类别IoU的平均
        return mIoU
    
    def meanIntersectionOverUnion1(self):
        IoU = self.IntersectionOverUnion()
        mIoU = np.nanmean(IoU) # 求各类别IoU的平均
        # mIoU = np.nanmean(IoU[1:]) # 求各类别IoU的平均
        return mIoU
    
    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表(A+B-AB) 
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU
    
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        diag_sum = np.diag(self.confusionMatrix).sum()
        total_sum = self.confusionMatrix.sum()
        # ignore specified background
        for IgLabel in self.ignore_labels:
            total_sum -= self.confusionMatrix[IgLabel].sum() 
            diag_sum -= self.confusionMatrix[IgLabel,IgLabel]
        acc = diag_sum / total_sum 
        return acc

    def Precision(self):
        # return precision
        # Pre = (TP) / TP + FP
        Pre = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return Pre # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def Recall(self):
        # return recall
        # Rec = (TP) / TP + FN
        Rec = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return Rec # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的Recall

if __name__ == '__main__':
    metric = SegmentationMetric(3,ignore_labels=[0])
    pre_mask = np.array([[0,2,0],[2,1,0],[1,2,1]])
    true_mask = np.array([[0,1,0],[2,1,0],[2,2,1]])
    metric.addBatch(pre_mask,true_mask)
    print(metric.confusionMatrix)
    print('IoU: ',metric.IntersectionOverUnion())
    print('mIoU: ',metric.meanIntersectionOverUnion())
    print('PA: ',metric.pixelAccuracy())
    print('Recall: ',metric.Recall())
    print('Precision:',metric.Precision())