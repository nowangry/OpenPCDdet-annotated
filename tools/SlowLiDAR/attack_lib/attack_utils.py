import torch
from shapely.geometry import Polygon
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.SlowLiDAR.dist_lib.set_distance import chamfer


def max_objects_loss(pred, conf_thres, eta=-1):
    # cls_pred = pred[0, ...]
    cls_pred = pred[0]['scores']
    thres_matrix = torch.full(cls_pred.shape, conf_thres).cuda()
    max_ten = torch.clamp(thres_matrix - cls_pred, min=eta)
    loss = torch.mean(max_ten)
    return loss


def box_overlap_loss(corners, eta=-1):
    if len(corners) == 0 or len(corners) == 1:
        return 0

    overlap_sum = 0
    for i in range(corners.shape[0]):
        x_i1 = corners[i, 0]
        x_i2 = corners[i, 1]
        x_i3 = corners[i, 2]
        x_i4 = corners[i, 3]
        x_i_center = (x_i1 + x_i2 + x_i3 + x_i4) / 4
        area_i = torch.linalg.norm(x_i1 - x_i2, ord=2) * torch.linalg.norm(x_i1 - x_i3, ord=2)
        for j in range(i + 1, corners.shape[0]):
            x_j1 = corners[j, 0]
            x_j2 = corners[j, 1]
            x_j3 = corners[j, 2]
            x_j4 = corners[j, 3]
            x_j_center = (x_j1 + x_j2 + x_j3 + x_j4) / 4
            area_j = torch.linalg.norm(x_j1 - x_j2, ord=2) * torch.linalg.norm(x_j1 - x_j3, ord=2)
            dist_ij = torch.linalg.norm(x_i_center - x_j_center, ord=2)
            max_ten = torch.clamp(torch.sqrt(area_i) * torch.sqrt(area_j) / dist_ij, min=eta)
            overlap_sum += max_ten

    nums = corners.shape[0] * (corners.shape[0] - 1) * 0.5
    loss = overlap_sum / nums

    return loss


def box_overlap_loss_3d(boxes_3d, eta=-1):
    if len(boxes_3d) == 0 or len(boxes_3d) == 1:
        return 0

    overlap_sum = 0
    for i in range(boxes_3d.shape[0]):
        x_i_center = boxes_3d[i, :3]
        area_i = boxes_3d[i, 3] * boxes_3d[i, 4]
        for j in range(i + 1, boxes_3d.shape[0]):
            x_j_center = boxes_3d[j, :3]
            area_j = boxes_3d[j, 3] * boxes_3d[j, 4]

            dist_ij = torch.linalg.norm(x_i_center - x_j_center, ord=2)
            max_ten = torch.clamp(torch.sqrt(area_i) * torch.sqrt(area_j) / dist_ij, min=eta)
            overlap_sum += max_ten

    nums = boxes_3d.shape[0] * (boxes_3d.shape[0] - 1) * 0.5
    loss = overlap_sum / nums

    return loss


def total_loss(pred, boxes_3d=None, cls_threshold=0.05, ori_data=None, adv_data=None):
    max_obj_loss = max_objects_loss(pred, cls_threshold)
    box_bound_loss = 0
    chamfer_dist = 0

    # enable box_bound_loss and chamfer_dist requires a powerful GPU with large memory
    if boxes_3d is not None:
        box_bound_loss = box_overlap_loss_3d(boxes_3d)
    if ori_data is not None and adv_data is not None:
        chamfer_dist = chamfer(ori_data, adv_data)

    alpha = 1.0
    beta = 0.1
    loss = max_obj_loss + alpha * box_bound_loss + beta * chamfer_dist
    # loss = max_obj_loss

    return loss
