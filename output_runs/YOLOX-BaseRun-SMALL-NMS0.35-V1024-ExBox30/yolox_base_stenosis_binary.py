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
        # Define yourself dataset path
        self.data_dir = "/ai/mnt/data/stenosis/selected_small/Binary/FOLD0/"
        self.train_ann = "train_binary.json"
        self.val_ann = "val_binary.json"
        self.test_ann = "val_binary.json"
        self.fold = "FOLD0"
        self.exp_name = "YOLOX-BaseRun-SMALL-NMS0.35-V1024-ExBox30"
        self.output_dir = os.path.join("/ai/mnt/code/YOLOX/output_runs", self.exp_name)
        self.num_classes = 1

        self.max_epoch = 200
        self.data_num_workers = 4
        self.eval_interval = 1
