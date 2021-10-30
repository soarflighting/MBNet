import numpy as np
import torch
import torch.nn.functional as F
from evalution_segmentaion import *

def calc_iou(out,gt_labels,n_classes):

    # 对像素值按通道做softmax
    out = F.log_softmax(out, dim=1)
    # 求每个通道的最大索引,并转换为numpy
    pre_labels = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_labels]

    true_label = gt_labels.squeeze(1).data.cpu().numpy()
    true_label = [i for i in true_label]

    # 计算混淆矩阵
    confusion = calc_semantic_segmentation_confusion(
        pre_label, true_label,n_classes)

    iou = calc_semantic_segmentation_iou(confusion)

    return iou



def eval_seg(out,gt_labels,n_classes):

    # 对像素值按通道做softmax
    out = F.log_softmax(out, dim=1)
    # 求每个通道的最大索引,并转换为numpy
    pre_labels = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_labels]

    true_label = gt_labels.squeeze(1).data.cpu().numpy()
    true_label = [i for i in true_label]

    result = eval_semantic_segmentation(pre_label,true_label,n_classes)

    return result['miou'],result['pixel_accuracy'],result['class_accuracy'][1],result['mean_class_accuracy']

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
