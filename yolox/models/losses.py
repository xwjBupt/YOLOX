#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from math import exp
from .ssim import SSIM_Loss
from termcolor import cprint

global_index = 0
# class IOUloss(nn.Module):
#     def __init__(self, reduction="none", loss_type="giou"):
#         super(IOUloss, self).__init__()
#         self.reduction = reduction
#         self.loss_type = loss_type

#     def forward(self, pred, target):
#         assert pred.shape[0] == target.shape[0]

#         pred = pred.view(-1, 4)
#         target = target.view(-1, 4)
#         tl = torch.max(
#             (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
#         )
#         br = torch.min(
#             (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
#         )

#         area_p = torch.prod(pred[:, 2:], 1)
#         area_g = torch.prod(target[:, 2:], 1)

#         en = (tl < br).type(tl.type()).prod(dim=1)
#         area_i = torch.prod(br - tl, 1) * en
#         area_u = area_p + area_g - area_i
#         iou = (area_i) / (area_u + 1e-16)

#         if self.loss_type == "iou":
#             loss = 1 - iou**2
#         elif self.loss_type == "giou":
#             c_tl = torch.min(
#                 (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
#             )
#             c_br = torch.max(
#                 (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
#             )
#             area_c = torch.prod(c_br - c_tl, 1)
#             giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
#             loss = 1 - giou.clamp(min=-1.0, max=1.0)
#         elif self.loss_type == "siou":
#             # Compute SIoU terms
#             # si = (
#             #     torch.min(
#             #         (pred[:, 0] + pred[:, 2] / 2), (target[:, 0] + target[:, 2] / 2)
#             #     )
#             #     - torch.max(
#             #         (pred[:, 0] - pred[:, 2] / 2), (target[:, 0] - target[:, 2] / 2)
#             #     )
#             #     + 1
#             # )
#             # sj = (
#             #     torch.min(
#             #         (pred[:, 1] + pred[:, 3] / 2), (target[:, 1] + target[:, 3] / 2)
#             #     )
#             #     - torch.max(
#             #         (pred[:, 1] - pred[:, 3] / 2), (target[:, 1] - target[:, 3] / 2)
#             #     )
#             #     + 1
#             # )
#             # s_union = (pred[:, 2] + 1) * (pred[:, 3] + 1) + (
#             #     pred[:, 1] + pred[:, 3] / 2 - target[:, 1] - target[:, 3] / 2
#             # ) * (pred[:, 0] + pred[:, 2] / 2 - target[:, 0] - target[:, 2] / 2)
#             # s_intersection = si * sj

#             # # Compute SCYLLA-IoU
#             # siou = iou - (s_intersection / s_union)
#             # loss = 1 - siou

#             ### >>>> ####
#             # from https://blog.csdn.net/weixin_43980331/article/details/126159134
#             # --------------------角度损失(Angle cost)------------------------------
#             gt_p_center_D_value_w = torch.abs(
#                 (target[:, 0] - pred[:, 0])
#             )  # 真实框和预测框中心点的宽度差
#             gt_p_center_D_value_h = torch.abs(
#                 (target[:, 1] - pred[:, 1])
#             )  # 真实框和预测框中心点的高度差
#             sigma = torch.pow(
#                 gt_p_center_D_value_w**2 + gt_p_center_D_value_h**2, 0.5
#             )  # 真实框和预测框中心点的距离
#             sin_alpha = torch.abs(gt_p_center_D_value_h) / (
#                 sigma + 1e-8
#             )  # 真实框和预测框中心点的夹角α
#             sin_beta = torch.abs(gt_p_center_D_value_w) / (
#                 sigma + 1e-8
#             )  # 真实框和预测框中心点的夹角β

#             threshold = pow(2, 0.5) / 2
#             # threshold = (
#             #     torch.pow(torch.tensor(2.0), 0.5) / 2
#             # )  # 夹角阈值 0.7071068 = sin45° = 二分之根二
#             # torch.where(condition，a，b)其中
#             # 输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。
#             sin_alpha = torch.where(
#                 sin_alpha < threshold, sin_beta, sin_alpha
#             )  # α小于45°则考虑优化β，否则优化α
#             angle_cost = torch.cos(2 * (torch.arcsin(sin_alpha) - math.pi / 4))

#             # -----------------距离损失(Distance cost)-----------------------------
#             # min_enclosing_rec_tl：最小外接矩形左上坐标
#             # min_enclosing_rec_br：最小外接矩形右下坐标
#             min_enclosing_rec_tl = torch.min(
#                 (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
#             )
#             min_enclosing_rec_br = torch.max(
#                 (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
#             )

#             # 最小外接矩形的宽高
#             min_enclosing_rec_br_w = (min_enclosing_rec_br - min_enclosing_rec_tl)[:, 0]
#             min_enclosing_rec_br_h = (min_enclosing_rec_br - min_enclosing_rec_tl)[:, 1]

#             # 真实框和预测框中心点的宽度(高度)差 / 以最小外接矩形的宽（高） 的平方
#             rho_x = (gt_p_center_D_value_w / min_enclosing_rec_br_w) ** 2
#             rho_y = (gt_p_center_D_value_h / min_enclosing_rec_br_h) ** 2

#             gamma = 2 - angle_cost
#             # 距离损失
#             distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

#             # ----------------形状损失(Shape cost)----------------------
#             w_pred = pred[:, 2]  # 预测框的宽
#             w_gt = target[:, 2]  # 真实框的宽
#             h_pred = pred[:, -1]  # 预测框的高
#             h_gt = target[:, -1]  # 真实框的高
#             # 预测框的宽 - 真实框的宽的绝对值 / 预测框的宽和真实框的宽中的最大值
#             omiga_w = torch.abs(w_pred - w_gt) / torch.max(w_pred, w_gt)
#             omiga_h = torch.abs(h_pred - h_gt) / torch.max(h_pred, h_gt)

#             # 作者使用遗传算法计算出θ接近4，因此作者定于θ参数范围为[2, 6]
#             theta = 4
#             # 形状损失
#             shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), theta) + torch.pow(
#                 1 - torch.exp(-1 * omiga_h), theta
#             )

#             # ------------------loss_siou----------------------------
#             siou = iou - 0.5 * (distance_cost + shape_cost)
#             loss = 1.0 - siou.clamp(min=-1.0, max=1.0)
#             if torch.isnan(loss).any():
#                 print(loss)

#             ### >>>> ####
#             # b1_x1, b1_y1, b1_x2, b1_y2 = box1
#             # b2_x1, b2_y1, b2_x2, b2_y2 = box2

#             # # IOU
#             # xx1 = np.maximum(b1_x1, b2_x1)
#             # yy1 = np.maximum(b1_y1, b2_y1)
#             # xx2 = np.minimum(b1_x2, b2_x2)
#             # yy2 = np.minimum(b1_y2, b2_y2)
#             # inter_w = np.maximum(0.0, xx2 - xx1)
#             # inter_h = np.maximum(0.0, yy2 - yy1)
#             # inter = inter_w * inter_h
#             # Union = (
#             #     (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#             #     + (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
#             #     - inter
#             # )
#             # IOU = inter / Union

#             # center_b_x = (b1_x1 + b1_x2) / 2
#             # center_b_y = (b1_y1 + b1_y2) / 2
#             # center_gtb_x = (b2_x1 + b2_x2) / 2
#             # center_gtb_y = (b2_y1 + b2_y2) / 2

#             # # ANGLE
#             # sigma = np.sqrt(
#             #     (center_gtb_x - center_b_x) ** 2 + (center_gtb_y - center_b_y) ** 2
#             # )
#             # lambda_ch = max(center_gtb_y, center_b_y) - min(center_gtb_y, center_b_y)
#             # lambda_x = lambda_ch / sigma
#             # angle = 1 - 2 * (np.sin(np.arctan(lambda_x) - np.pi / 4) ** 2)

#             # # DISTANCE
#             # lambda_cw = max(center_gtb_x, center_b_x) - min(center_gtb_x, center_b_x)
#             # Rho_x = ((center_gtb_x - center_b_x) / lambda_cw) ** 2
#             # Rho_y = ((center_gtb_y - center_b_y) / lambda_ch) ** 2
#             # gamma = 2 - angle
#             # Delat = (1 - np.exp(-1 * gamma * Rho_x)) + (1 - np.exp(-1 * gamma * Rho_y))

#             # # SHAPE
#             # Theta = 4
#             # pred_w = b1_y2 - b1_y1
#             # pred_h = b1_x2 - b1_x1
#             # gt_w = b2_y2 - b2_y1
#             # gt_h = b2_x2 - b2_x1
#             # Omega_w = abs(pred_w - gt_w) / max(pred_w, gt_w)
#             # Omega_h = abs(pred_h - gt_h) / max(pred_h, gt_h)
#             # Omega = (1 - np.exp(-1 * Omega_w)) ** Theta + (
#             #     1 - np.exp(-1 * Omega_h)
#             # ) ** Theta

#             # SIOU = 1 - IOU + (Delat + Omega) / 2

#         else:
#             assert False, "IOU Loss type {} not support yet"

#         if self.reduction == "mean":
#             loss = loss.mean()
#         elif self.reduction == "sum":
#             loss = loss.sum()

#         return loss


# class IOUloss:
#     """Calculate IoU loss."""

#     """
#     take from https://github.com/meituan/YOLOv6/blob/main/yolov6/utils/figure_iou.py#L75
#     """

#     def __init__(self, box_format="xywh", loss_type="siou", reduction="none", eps=1e-7):
#         """Setting of the class.
#         Args:
#             box_format: (string), must be one of 'xywh' or 'xyxy'.
#             iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
#             reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
#             eps: (float), a value to avoid divide by zero error.
#         """
#         self.box_format = box_format
#         self.iou_type = loss_type.lower()
#         self.reduction = reduction
#         self.eps = eps

#     def __call__(self, box1, box2):
#         """calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4]."""
#         if box1.shape[0] != box2.shape[0]:
#             box2 = box2.T
#             if self.box_format == "xyxy":
#                 b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#                 b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#             elif self.box_format == "xywh":
#                 b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
#                 b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
#                 b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
#                 b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
#         else:
#             if self.box_format == "xyxy":
#                 b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
#                 b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

#             elif self.box_format == "xywh":
#                 b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
#                 b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
#                 b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
#                 b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
#                 b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
#                 b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

#         # Intersection area
#         inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
#             torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
#         ).clamp(0)

#         # Union Area
#         w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
#         w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
#         union = w1 * h1 + w2 * h2 - inter + self.eps
#         iou = inter / union

#         cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
#         ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
#         if self.iou_type == "giou":
#             c_area = cw * ch + self.eps  # convex area
#             iou = iou - (c_area - union) / c_area
#         elif self.iou_type in ["diou", "ciou"]:
#             c2 = cw**2 + ch**2 + self.eps  # convex diagonal squared
#             rho2 = (
#                 (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
#                 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
#             ) / 4  # center distance squared
#             if self.iou_type == "diou":
#                 iou = iou - rho2 / c2
#             elif self.iou_type == "ciou":
#                 v = (4 / math.pi**2) * torch.pow(
#                     torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
#                 )
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + self.eps))
#                 iou = iou - (rho2 / c2 + v * alpha)
#         elif self.iou_type == "siou":
#             # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
#             s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
#             s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
#             sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)
#             sin_alpha_1 = torch.abs(s_cw) / sigma
#             sin_alpha_2 = torch.abs(s_ch) / sigma
#             threshold = pow(2, 0.5) / 2
#             sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
#             angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
#             rho_x = (s_cw / cw) ** 2
#             rho_y = (s_ch / ch) ** 2
#             gamma = angle_cost - 2
#             distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
#             omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
#             omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
#             shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
#                 1 - torch.exp(-1 * omiga_h), 4
#             )
#             iou = iou - 0.5 * (distance_cost + shape_cost)
#         loss = 1.0 - iou

#         if self.reduction == "sum":
#             loss = loss.sum()
#         elif self.reduction == "mean":
#             loss = loss.mean()

#         return loss


class IOUloss(nn.Module):
    """Calculate IoU loss."""

    """
    take from https://github.com/meituan/YOLOv6/blob/main/yolov6/utils/figure_iou.py#L75
    """

    def __init__(self, box_format="xywh", loss_type="siou", reduction="none", eps=1e-7):
        """Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        super(IOUloss, self).__init__()
        self.box_format = box_format
        self.iou_type = loss_type.lower()
        self.reduction = reduction
        self.eps = eps

    def forward(self, box1, box2):
        """calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4]."""
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T
            if self.box_format == "xyxy":
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            elif self.box_format == "xywh":
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            if self.box_format == "xyxy":
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

            elif self.box_format == "xywh":
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
                b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
                b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
                b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
            torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
        ).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == "giou":
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ["diou", "ciou"]:
            c2 = cw**2 + ch**2 + self.eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if self.iou_type == "diou":
                iou = iou - rho2 / c2
            elif self.iou_type == "ciou":
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == "siou":
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
                1 - torch.exp(-1 * omiga_h), 4
            )
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


def pairwise_bbox_iou(box1, box2, box_format="xywh"):
    """Calculate iou.
    This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
    """
    if box_format == "xyxy":
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        area_1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
        area_2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)

    elif box_format == "xywh":
        lt = torch.max(
            (box1[:, None, :2] - box1[:, None, 2:] / 2),
            (box2[:, :2] - box2[:, 2:] / 2),
        )
        rb = torch.min(
            (box1[:, None, :2] + box1[:, None, 2:] / 2),
            (box2[:, :2] + box2[:, 2:] / 2),
        )

        area_1 = torch.prod(box1[:, 2:], 1)
        area_2 = torch.prod(box2[:, 2:], 1)
    valid = (lt < rb).type(lt.type()).prod(dim=2)
    inter = torch.prod(rb - lt, 2) * valid
    return inter / (area_1[:, None] + area_2 - inter)


class IOU_SSIM(nn.Module):
    """Calculate IoU loss."""

    """
    take from https://github.com/meituan/YOLOv6/blob/main/yolov6/utils/figure_iou.py#L75
    """

    def __init__(
        self,
        box_format="xywh",
        loss_type="siou",
        reduction="none",
        eps=1e-7,
        cal_thresh=0.3,
        size=(32, 32),
        **kwargs
    ):
        """Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        super(IOU_SSIM, self).__init__()
        self.box_format = box_format
        self.iou_type = loss_type.lower()
        self.reduction = reduction
        self.eps = eps
        self.cal_thresh = cal_thresh
        self.size = size
        self.ssim = SSIM_Loss(in_channels=3)
        self.mse = nn.L1Loss(size_average=None, reduce=None, reduction="none")

    def forward(
        self,
        batch_idx_pred_targets: list,
        batch_idx_reg_targets: list,
        imgs: torch.Tensor,
    ):
        assert len(batch_idx_pred_targets) == len(
            batch_idx_reg_targets
        ), "batch do not match"
        assert len(batch_idx_pred_targets) == imgs.shape[0], "batch do not match"
        pred_img_batches = []
        reg_img_batches = []
        Loss = 0
        batches = len(batch_idx_pred_targets)
        for i in range(batches):  # TODO
            batch_idx_pred_target = batch_idx_pred_targets[i]
            batch_idx_reg_target = batch_idx_reg_targets[i]
            for box1 in batch_idx_reg_target:
                for box2 in batch_idx_pred_target:
                    if box2[3] > 2 and box2[2] > 2:
                        iou = self.get_two_box_iou(box1, box2)
                        if iou > self.cal_thresh:
                            gt_patch = imgs[i][
                                :,
                                int(box1[1]) : int(box1[1] + box1[3]),
                                int(box1[0]) : int(box1[0] + box1[2]),
                            ]
                            pred_patch = imgs[i][
                                :,
                                int(box2[1]) : int(box2[1] + box2[3]),
                                int(box2[0]) : int(box2[0] + box2[2]),
                            ]
                            try:
                                if self.size:
                                    gt_patch = F.interpolate(
                                        gt_patch.unsqueeze(0),
                                        size=self.size,
                                        mode="bilinear",
                                    )
                                    pred_patch = F.interpolate(
                                        pred_patch.unsqueeze(0),
                                        size=self.size,
                                        mode="bilinear",
                                    )
                                pred_img_batches.append(pred_patch)
                                reg_img_batches.append(gt_patch)
                            except Exception as e:
                                cprint(
                                    "GT box as {} -- Pred box as {}".format(box1, box2),
                                    color="yellow",
                                )
                                continue
                        else:
                            continue
        if len(pred_img_batches) > 0:
            pred_tensor = torch.cat(pred_img_batches, dim=0)
            gt_tensor = torch.cat(reg_img_batches, dim=0)
            loss = self.mse(pred_tensor / 255, gt_tensor / 255) + self.ssim(
                pred_tensor, gt_tensor
            )
            Loss = Loss + loss
        else:
            Loss = Loss + 0
        if isinstance(Loss, int):
            return 0
        else:
            return Loss.mean()

    def get_two_box_iou(self, box1, box2):
        tl = torch.max((box1[:2] - box1[2:] / 2), (box2[:2] - box2[2:] / 2))
        br = torch.min((box1[:2] + box1[2:] / 2), (box2[:2] + box2[2:] / 2))

        area_p = torch.prod(box1[2:], 0)
        area_g = torch.prod(box2[2:], 0)

        en = (tl < br).type(tl.type()).prod(dim=0)
        area_i = torch.prod(br - tl, 0) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)
        return iou
