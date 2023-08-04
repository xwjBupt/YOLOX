#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        MODELNAME = "YOLOX#L"
        if MODELNAME == "YOLOX":
            self.depth = 1.33
            self.width = 1.25
            self.data_num_workers = 4
            self.batch_size = 2
            self.ckpt = "/ai/mnt/code/YOLOX/yolox/models/yolox.pth"
        elif MODELNAME == "YOLOX#L":
            self.depth = 1.0
            self.width = 1.0
            self.data_num_workers = 6
            self.batch_size = 3
            self.ckpt = "/ai/mnt/code/YOLOX/yolox/models/yolox_l.pth"
        elif MODELNAME == "YOLOX#M":
            self.depth = 0.67
            self.width = 0.75
            self.data_num_workers = 8
            self.batch_size = 4
            self.ckpt = "/ai/mnt/code/YOLOX/yolox/models/yolox_m.pth"
        elif MODELNAME == "YOLOX#S":
            self.depth = 0.33
            self.width = 0.50
            self.data_num_workers = 12
            self.batch_size = 6
        elif MODELNAME == "YOLOX#T":
            self.depth = 0.33
            self.width = 0.375
            self.input_size = (416, 416)
            self.mosaic_scale = (0.5, 1.5)
            self.random_size = (10, 20)
            self.test_size = (416, 416)
        else:
            assert False, "{} do not support yet"
        self.nmsthre = 0.35
        self.input_size = (1024, 1024)
        self.test_size = (1024, 1024)
        self.multiscale_range = 8
        self.crop_dict = dict(
            width=256,
            height=256,
            erosion_rate=0,
            min_area=64,
            min_visibility=0,
            format="coco",
            random_rate=0.5,
        )
        self.zoom_blur_dict = dict(
            blur_limit=7, allow_shifted=True, always_apply=False, p=0.35
        )
        self.motion_blur_dict = dict(
            max_factor=1.31, step_factor=(0.01, 0.03), always_apply=False, p=0.4
        )
        self.cut_copy_dict = dict(
            iou_thresh=0.2, paste_number=10, thresh=64, expand=5, p=0.25
        )
        self.clip_dict = dict(low=48, high=192, p=0.25)
        # Define yourself dataset path
        self.data_dir = "/ai/mnt/data/stenosis/selected/Degree/FOLD0/"
        self.train_ann = "train_degree.json"
        self.val_ann = "val_degree.json"
        self.test_ann = "val_degree.json"
        self.fold = "FOLD0"
        self.cal_thresh = 0.1
        self.ssim_size = (32, 32)
        self.exp_name = (
            "%s-DEGREE-NMS0.35-V1024-SR8-CROP0.5_first_256-ZOOM0.35-MOTION.04-tf5e_3-cutcopy_ex5-clip0.25-siou-fixed_iou_similarity0.1_V32"
            % MODELNAME
        )
        self.output_dir = os.path.join(
            "/ai/mnt/code/YOLOX/output_runs/Degree", self.exp_name
        )
        self.num_classes = 3
        self.max_epoch = 100
        self.test_conf = 0.005
        self.eval_interval = 1
        self.print_interval = 150
        self.iou_type = "siou"
        self.box_contain_thresh = 0.1
        self.use_cab = False

        # self.basic_lr_per_img = 0.01 / 16.0  # 0.01 / 64.0
