############################################
##
## Evaluation Metrics
##
## Author:  Peizhi Yan
##   Date:  Mar. 7, 2021
##
############################################

import numpy as np

def IOU(targ, pred):
    """ Compute the Intersection over Union """
    # targ: the target mask (binary array)
    # pred: the predicted mask (binary array)
    intersection = np.logical_and(targ, pred)
    union = np.logical_or(targ, pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def AP(targ, pred):
    """ Compute the Average Precision """
    # targ: the target mask (binary array)
    # pred: the predicted mask (binary array)
    TP = np.sum(np.logical_and(targ, pred))
    FP = np.sum(pred) - TP
    ap = TP / (TP + FP)
    return ap

def mIOU(targs, preds):
    """ Compute the mean IOU over multiple target-prediction pairs """
    return IOU(targs, preds)

def mAP(targs, preds):
    """ Compute the mean IOU over multiple target-prediction pairs """
    return AP(targs, preds)






