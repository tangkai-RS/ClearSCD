import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class BCDEvaluator(object):
    '''only see change class'''
    def __init__(self, num_class=2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) # row is True col is pred
        self.pre_cal = False
        
    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Precision(self): # precision
        self._pre_cal() if not self.pre_cal else 0
        pre = self.TP + self.FP
        pre = np.where(pre==0, 0, self.TP / pre)
        pre = np.round(pre, 5)
        return pre[-1]

    def Recall(self): # recall
        self._pre_cal() if not self.pre_cal else 0
        rec = self.TP + self.FN
        rec = np.where(rec==0, 0, self.TP / rec)
        rec = np.round(rec, 5)
        return rec[-1]
    
    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP[1] + self.FN[1] + self.FP[1]
        IoU = np.where(IoU==0, 0, self.TP[1] / IoU)
        IoU = np.round(np.nanmean(IoU), 5)
        return IoU
    
    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2* self.TP + self.FP + self.FN)
        F1 = np.round(F1, 5)
        return F1[-1]
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False
        
        
class SEGEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) # row is True col is pred
        self.pre_cal = False
        
    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Precision(self): # precision
        self._pre_cal() if not self.pre_cal else 0
        pre = self.TP + self.FP
        pre = np.where(pre==0, 0, self.TP / pre)
        pre = np.round(pre, 5)
        return pre

    def Recall(self): # recall
        self._pre_cal() if not self.pre_cal else 0
        rec = self.TP + self.FN
        rec = np.where(rec==0, 0, self.TP / rec)
        rec = np.round(rec, 5)
        return rec

    def Mean_Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        MIoU = self.TP + self.FN + self.FP
        MIoU = np.where(MIoU==0, 0, self.TP / MIoU)
        MIoU = np.round(np.nanmean(MIoU), 5)
        return MIoU
    
    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP + self.FN + self.FP
        IoU = np.where(IoU==0, 0, self.TP / IoU)
        return IoU
    
    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2* self.TP + self.FP + self.FN)
        F1 = np.round(F1, 5)
        return F1
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False


class SCD_NoChange_Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) # row is True col is pred
        self.pre_cal = False
        self.class_list = []
        
    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Precision(self): # precision
        self._pre_cal() if not self.pre_cal else 0
        pre = self.TP + self.FP
        pre = np.where(pre==0, 0, self.TP / pre)
        pre = np.round(pre, 5)
        return pre

    def Recall(self): # recall
        self._pre_cal() if not self.pre_cal else 0
        rec = self.TP + self.FN
        rec = np.where(rec==0, 0, self.TP / rec)
        rec = np.round(rec, 5)
        return rec

    def MIoU_pre(self):
        self._pre_cal() if not self.pre_cal else 0
        MIoU = self.TP + self.FN + self.FP
        MIoU = np.where(MIoU==0, 0, self.TP / MIoU)
        self.MIoU_list = MIoU
    
    def Mean_Intersection_over_Union(self):
        n = len(np.unique(self.class_list)) - 1 # 减1去掉未标注类别
        MIoU = np.round(np.nansum(self.MIoU_list) / n, 5)
        return MIoU
    
    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP + self.FN + self.FP
        IoU = np.where(IoU==0, 0, self.TP / IoU)
        return IoU
    
    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2* self.TP + self.FP + self.FN)
        F1 = np.round(F1, 5)
        return F1
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.class_list.extend(np.unique(gt_image))

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False      
        

class SCD_Change_Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) # row is True col is pred
        self.pre_cal = False
        self.class_list = []
        
    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Precision(self): # precision
        self._pre_cal() if not self.pre_cal else 0
        pre = self.TP + self.FP
        pre = np.where(pre==0, 0, self.TP / pre)
        pre = np.round(pre, 5)
        return pre

    def Recall(self): # recall
        self._pre_cal() if not self.pre_cal else 0
        rec = self.TP + self.FN
        rec = np.where(rec==0, 0, self.TP / rec)
        rec = np.round(rec, 5)
        return rec
    
    def Mean_Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        MIoU = self.TP + self.FN + self.FP
        MIoU = np.where(MIoU==0, 0, self.TP / MIoU)
        n = len(np.unique(self.class_list)) - 1 # 减1去掉未标注类别
        MIoU = np.round(np.nansum(MIoU) / n, 5)
        return MIoU
    
    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP + self.FN + self.FP
        IoU = np.where(IoU==0, 0, self.TP / IoU)
        return IoU
    
    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2* self.TP + self.FP + self.FN)
        F1 = np.round(F1, 5)
        return F1
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.class_list.extend(np.unique(gt_image))

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False    
          
if __name__ == '__main__':
    pass
    
