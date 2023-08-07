#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from .ssim import SSIM_Loss
from termcolor import cprint


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
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

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


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
        if self.cal_thresh >= 1:
            return 0
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
            H, W = imgs[i].shape[-2], imgs[i].shape[-1]
            for box1 in batch_idx_reg_target:
                for box2 in batch_idx_pred_target:
                    if box2[3] > 2 and box2[2] > 2:
                        iou = self.get_two_box_iou(box1, box2)
                        if iou > self.cal_thresh:
                            try:
                                gt_patch = imgs[i][
                                    :,
                                    max(0, int(box1[1] - box1[3] / 2)) : min(
                                        int(box1[1] + box1[3] / 2), H
                                    ),
                                    max(0, int(box1[0] - box1[2] / 2)) : min(
                                        int(box1[0] + box1[2] / 2), W
                                    ),
                                ]
                                pred_patch = imgs[i][
                                    :,
                                    max(0, int(box2[1] - box2[3] / 2)) : min(
                                        int(box2[1] + box2[3] / 2), H
                                    ),
                                    max(0, int(box2[0] - box2[2] / 2)) : min(
                                        int(box2[0] + box2[2] / 2), W
                                    ),
                                ]

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
                                    "GT box as {} -- Pred box as {} in format cxcywh, gt shape with {} pred shape with {} Exception as {}".format(
                                        box1, box2, gt_patch.shape, pred_patch.shape, e
                                    ),
                                    color="yellow",
                                )
                                continue
                            # img = imgs[0][0].detach().clone().cpu().numpy().astype(np.uint8)
                            # cv2.imwrite('/ai/mnt/code/YOLOX/raw.png',img)
                            # cv2.rectangle(img,(int(box1[0] - box1[2] / 2),int(box1[1] - box1[3] / 2)),(int(box1[0] + box1[2] / 2),int(
                            #            box1[1] + box1[3] / 2
                            #        )),color = 45,thickness = 4)

                            # cv2.rectangle(
                            #    img,
                            #     (
                            #         int(box2[0] - box2[2] / 2),
                            #         int(box2[1] - box2[3] / 2),
                            #     ),
                            #     (
                            #         int(box2[0] + box2[2] / 2),
                            #         int(box2[1] + box2[3] / 2),
                            #     ),
                            #     color=3,
                            #     thickness=4,
                            # )

                            # cv2.rectangle(
                            #    img,
                            #     (
                            #         int(box2[1]),
                            #         int(box2[0]),
                            #     ),
                            #     (
                            #         int(box2[2]),
                            #         int(box2[3]),
                            #     ),
                            #     color=3,
                            #     thickness=4,
                            # )

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
