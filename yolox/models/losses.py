#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="giou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "siou":
            # Compute SIoU terms
            si = (
                torch.min(pred[:, 2], target[:, 2])
                - torch.max(pred[:, 0], target[:, 0])
                + 1
            )
            sj = (
                torch.min(pred[:, 3], target[:, 3])
                - torch.max(pred[:, 1], target[:, 1])
                + 1
            )
            s_union = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1) + (
                pred[:, 2] - target[:, 0] + 1
            ) * (pred[:, 3] - target[:, 1] + 1)
            s_intersection = si * sj

            # Compute SCYLLA-IoU
            siou = iou - (s_intersection / s_union)
            loss = 1 - siou
        else:
            assert False, "IOU Loss type {} not support yet"
        # elif self.loss_type == "wiou":
        #     intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(
        #         y2 - y1, min=0
        #     )
        #     pred_boxes_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
        #         pred_boxes[:, 3] - pred_boxes[:, 1]
        #     )
        #     target_boxes_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (
        #         target_boxes[:, 3] - target_boxes[:, 1]
        #     )
        #     union_area = pred_boxes_area + target_boxes_area - intersection_area
        #     iou = intersection_area / union_area
        #     loss = 1 - iou.mean()

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
