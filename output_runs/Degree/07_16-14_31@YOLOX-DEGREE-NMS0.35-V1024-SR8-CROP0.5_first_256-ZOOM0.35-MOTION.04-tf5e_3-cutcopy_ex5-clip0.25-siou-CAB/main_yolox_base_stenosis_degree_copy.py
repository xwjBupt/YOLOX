#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
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
        self.exp_name = "YOLOX-DEGREE-NMS0.35-V1024-SR8-CROP0.5_first_256-ZOOM0.35-MOTION.04-tf5e_3-cutcopy_ex5-clip0.25-siou-CAB"
        self.output_dir = os.path.join(
            "/ai/mnt/code/YOLOX/output_runs/Degree", self.exp_name
        )
        self.num_classes = 3
        self.max_epoch = 100
        self.data_num_workers = 4
        self.batch_size = 2
        self.test_conf = 0.005
        self.eval_interval = 1
        self.print_interval = 150
        self.iou_type = "siou"
        self.box_contain_thresh = 0.1
        self.use_cab = True
