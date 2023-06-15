#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import sys
import random
import time
import warnings
from loguru import logger
import argparse
import torch
import torch.backends.cudnn as cudnn

from yolox.exp import get_exp
from yolox.core import Trainer
from yolox.utils import configure_module, configure_omp


class AssignVisualizer(Trainer):
    def __init__(self, exp, args):
        super().__init__(exp, args)
        self.batch_cnt = 0
        self.vis_dir = os.path.join(self.file_name, "AssignVisualizer")
        os.makedirs(self.vis_dir, exist_ok=True)

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            path_prefix = os.path.join(self.vis_dir, f"assign_vis_{self.batch_cnt}_")
            self.model.visualize(inps, targets, path_prefix)

        if self.use_model_ema:
            self.ema_model.update(self.model)

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
        )
        self.batch_cnt += 1
        if self.batch_cnt >= self.args.max_batch:
            sys.exit(0)

    def after_train(self):
        logger.info("Finish visualize assignment, exit...")


def assign_vis_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="weather in debug mode, given = no debug, not given  = in debug",
    )
    parser.add_argument("--gitinfo", type=str, default="<<< DEBUG >>>")
    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=0, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/ai/mnt/code/YOLOX/output_runs/YOLOX-BaseRun-SMALL-NMS0.35-V1024-ExBox30/yolox_base_stenosis_binary.py",
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str,
        help="checkpoint file",
    )
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        default="wandb",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--max-batch", type=int, default=100, help="max batch of images to visualize"
    )

    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_omp()
    cudnn.benchmark = True
    visualizer = AssignVisualizer(exp, args)
    visualizer.train()


if __name__ == "__main__":
    configure_module()
    args = assign_vis_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    if not args.ckpt:
        args.ckpt = os.path.join(os.path.dirname(args.exp_file), "best_ckpt.pth")
    main(exp, args)
