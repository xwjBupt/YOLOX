#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""
import albumentations as A
import math
import random
import copy

import cv2
import numpy as np
from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(
        img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img
    )  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(
                value
            )
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def issmallobject(bbox, thresh):
    if bbox[0] * bbox[1] <= thresh:
        return True
    else:
        return False


def norm_sampling(search_space):
    # 随机生成点
    search_x_left, search_y_left, search_x_right, search_y_right = search_space
    try:
        new_bbox_x_center = random.randint(int(search_x_left), int(search_x_right))
        new_bbox_y_center = random.randint(int(search_y_left), int(search_y_right))
    except:
        print("this is at here norm_sampling with {}".format(search_space))
    return [new_bbox_x_center, new_bbox_y_center]


def flip_bbox(roi):
    roi = roi[:, ::-1, :]
    return roi


def sampling_new_bbox_center_point(img_shape, bbox):
    #### sampling space ####
    height, width, nc = img_shape
    cl, x_left, y_left, x_right, y_right = bbox
    bbox_w, bbox_h = x_right - x_left, y_right - y_left
    ### left top ###
    if x_left <= width / 2:
        search_x_left, search_y_left, search_x_right, search_y_right = (
            width * 0.6,
            height / 2,
            width * 0.75,
            height * 0.75,
        )
    if x_left > width / 2:
        search_x_left, search_y_left, search_x_right, search_y_right = (
            width * 0.25,
            height / 2,
            width * 0.5,
            height * 0.75,
        )
    return [search_x_left, search_y_left, search_x_right, search_y_right]


def bbox_iou(box1, box2):
    cl, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    cl, b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:  # strong condition
        inter_area = inter_width * inter_height
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou


def random_add_patches(bbox, rescale_boxes, shape, paste_number, iou_thresh, enlarge=0):
    temp = []
    for rescale_bbox in rescale_boxes:
        temp.append(rescale_bbox)
    cl, x_left, y_left, x_right, y_right = bbox
    bbox_w, bbox_h = x_right - x_left, y_right - y_left
    center_search_space = sampling_new_bbox_center_point(shape, bbox)
    # height, width
    success_num = 0
    new_bboxes = []
    while success_num < paste_number:
        new_bbox_x_center, new_bbox_y_center = norm_sampling(center_search_space)
        # print(norm_sampling(center_search_space))
        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = (
            new_bbox_x_center - 0.5 * bbox_w - enlarge,
            new_bbox_y_center - 0.5 * bbox_h - enlarge,
            new_bbox_x_center + 0.5 * bbox_w + enlarge,
            new_bbox_y_center + 0.5 * bbox_h + enlarge,
        )
        new_bbox = [
            cl,
            int(new_bbox_x_left),
            int(new_bbox_y_left),
            int(new_bbox_x_right),
            int(new_bbox_y_right),
        ]
        ious = [bbox_iou(new_bbox, bbox_t) for bbox_t in rescale_boxes]
        if max(ious) <= iou_thresh and (
            new_bbox_x_right < shape[1] and new_bbox_y_right < shape[0]
        ):
            # print(
            #     "new_bbox_x_right {} width {} -- new_bbox_y_right {} height {}".format(
            #         new_bbox_x_right, shape[1], new_bbox_y_right, shape[0]
            #     )
            # )
            # for bbox_t in rescale_boxes:
            # iou =  bbox_iou(new_bbox[1:],bbox_t[1:])
            # if(iou <= iou_thresh):
            success_num += 1
            temp.append(new_bbox)
            new_bboxes.append(new_bbox)
        else:
            continue
    return new_bboxes


def rescale_yolo_labels(labels, img_shape):
    height, width, nchannel = img_shape
    rescale_boxes = []
    for box in list(labels):
        x_c = float(box[1]) * width
        y_c = float(box[2]) * height
        w = float(box[3]) * width
        h = float(box[4]) * height
        x_left = x_c - w * 0.5
        y_left = y_c - h * 0.5
        x_right = x_c + w * 0.5
        y_right = y_c + h * 0.5
        rescale_boxes.append(
            [box[0], int(x_left), int(y_left), int(x_right), int(y_right)]
        )
    return rescale_boxes


def copysmallobjects(
    image,
    labels,
):
    if len(labels) == 0:
        return
    rescale_labels = rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    all_boxes = []

    for idx, rescale_label in enumerate(rescale_labels):
        all_boxes.append(rescale_label)
        # 目标的长宽
        rescale_label_height, rescale_label_width = (
            rescale_label[4] - rescale_label[2],
            rescale_label[3] - rescale_label[1],
        )

        if (
            issmallobject((rescale_label_height, rescale_label_width), thresh=64 * 64)
            and rescale_label[0] == "1"
        ):
            roi = image[
                rescale_label[2] : rescale_label[4], rescale_label[1] : rescale_label[3]
            ]

            new_bboxes = random_add_patches(
                rescale_label,
                rescale_labels,
                image.shape,
                paste_number=2,
                iou_thresh=0.2,
            )
            count = 0

            # 将新生成的位置加入到label,并在相应位置画出物体
            for new_bbox in new_bboxes:
                count += 1
                all_boxes.append(new_bbox)
                cl, bbox_left, bbox_top, bbox_right, bbox_bottom = (
                    new_bbox[0],
                    new_bbox[1],
                    new_bbox[2],
                    new_bbox[3],
                    new_bbox[4],
                )
                try:
                    if count > 1:
                        roi = flip_bbox(roi)
                    image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                except ValueError:
                    continue

    # dir_name = find_str(image_dir)
    # save_dir = join(save_base_dir, dir_name)
    # check_dir(save_dir)
    # yolo_txt_dir = join(save_dir, basename(image_dir.replace(".jpg", "_augment.txt")))
    # cv2.imwrite(
    #     join(save_dir, basename(image_dir).replace(".jpg", "_augment.jpg")), image
    # )
    # convert_all_boxes(image.shape, all_boxes, yolo_txt_dir)


class CUTCOPY(object):
    def __init__(self, iou_thresh=0.2, paste_number=4, thresh=64, p=0.22, **kwargs):
        self.iou_thresh = iou_thresh
        self.paste_number = paste_number
        self.thresh = thresh * thresh
        self.p = p

    def __call__(self, image, labels, **kwargs):
        raw_img = copy.deepcopy(image)
        if random.random() < self.p:
            ## labels: [[x1,y1,x2,y2,label], ...]
            if len(labels) == 0:
                return
            rescale_labels = self.rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
            all_boxes = []
            for b in rescale_labels:
                raw_img = cv2.rectangle(
                    raw_img,
                    (int(b[1]), int(b[2])),  # (x,y)
                    (int(b[3]), int(b[4])),
                    color=(214, 128, 98),
                    thickness=2,
                )
                raw_img = cv2.putText(
                    raw_img,
                    "GT",
                    (int(b[1]), int(b[2])),
                    cv2.FONT_ITALIC,
                    1,
                    (76, 255, 0),
                    1,
                )

            for idx, rescale_label in enumerate(rescale_labels):
                all_boxes.append(rescale_label)
                # 目标的长宽
                rescale_label_height, rescale_label_width = (
                    rescale_label[4] - rescale_label[2],
                    rescale_label[3] - rescale_label[1],
                )

                if issmallobject(
                    (rescale_label_height, rescale_label_width), thresh=self.thresh
                ):
                    roi = image[
                        rescale_label[2] : rescale_label[4],
                        rescale_label[1] : rescale_label[3],
                    ]

                    new_bboxes = random_add_patches(
                        rescale_label,
                        rescale_labels,
                        image.shape,
                        paste_number=self.paste_number,
                        iou_thresh=self.iou_thresh,
                    )
                    count = 0

                    # 将新生成的位置加入到label,并在相应位置画出物体
                    for new_bbox in new_bboxes:
                        count += 1
                        cl, bbox_left, bbox_top, bbox_right, bbox_bottom = (
                            new_bbox[0],
                            new_bbox[1],
                            new_bbox[2],
                            new_bbox[3],
                            new_bbox[4],
                        )
                        try:
                            if random.random() < self.p:
                                roi = flip_bbox(roi)
                            image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                            all_boxes.append(new_bbox)
                            raw_img = cv2.rectangle(
                                raw_img,
                                (bbox_left, bbox_top),  # (x,y)
                                (bbox_right, bbox_bottom),
                                color=(255, 128, 0),
                                thickness=2,
                            )

                            raw_img = cv2.putText(
                                raw_img,
                                str(cl),
                                (bbox_left, bbox_top),
                                cv2.FONT_ITALIC,
                                1,
                                (0, 255, 0),
                                1,
                            )
                        except ValueError:
                            continue
            labels = self.reverse_rescale_yolo_labels(all_boxes, image.shape)
        # cv2.imwrite(
        #     '/ai/mnt/code/YOLOX/imgraw-txt.png', raw_img
        # )
        return image, np.array(labels)

    def reverse_rescale_yolo_labels(self, labels, img_shape):
        rescale_boxes = []
        height, width, nchannel = img_shape
        for box in list(labels):
            rescale_boxes.append([box[1], box[2], box[3], box[4], box[0]])
        return rescale_boxes

    def rescale_yolo_labels(self, labels, img_shape):
        height, width, nchannel = img_shape
        rescale_boxes = []
        for box in list(labels):
            if max(box) < 1:
                x_left = float(box[0]) * width
                y_left = float(box[1]) * height
                x_right = float(box[2]) * width
                y_right = float(box[3]) * height
                label = box[-1]
            else:
                x_left = box[0]
                y_left = box[1]
                x_right = box[2]
                y_right = box[3]
                label = box[-1]
            rescale_boxes.append(
                [label, int(x_left), int(y_left), int(x_right), int(y_right)]
            )
        return rescale_boxes


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, category_ids=[0]):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.category_ids = category_ids
        # self.transform = A.Compose(
        #     [
        #         A.HorizontalFlip(p=0.5),
        #         A.RandomBrightnessContrast(p=0.2),
        #         A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2),
        #     ],
        #     bbox_params=A.BboxParams(
        #         format="coco",
        #         min_area=384,
        #         min_visibility=0.1,
        #         label_fields=["category_ids"],
        #     ),
        # )

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img -= np.array(
                [144.5754766729963, 144.5754766729963, 144.5754766729963]
            ).reshape(3, 1, 1)
            img /= np.array(
                [55.8710224233549, 55.8710224233549, 55.8710224233549]
            ).reshape(3, 1, 1)
            img /= 255.0
        return img, np.zeros((1, 5))
