#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger
from git import Repo
import time
import torch
import torch.backends.cudnn as cudnn
import glob
import os
import setproctitle

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

# import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def git_commit(
    work_dir,
    commit_info,
    levels=10,
    postfixs=[".py", ".sh"],
    debug=False,
):
    cid = "not generate"
    branch = "master"
    if not debug:
        repo = Repo(work_dir)
        toadd = []
        branch = repo.active_branch.name
        for i in range(levels):
            for postfix in postfixs:
                filename = glob.glob(work_dir + (i + 1) * "/*" + postfix)
                for x in filename:
                    if (
                        not ("play" in x)
                        and not ("local" in x)
                        and not ("Untitled" in x)
                        and not ("wandb" in x)
                    ):
                        toadd.append(x)
        index = repo.index  # 获取暂存区对象
        index.add(toadd)
        index.commit(commit_info)
        cid = repo.head.commit.hexsha

    commit_tag = (
        commit_info
        + "\n"
        + "COMMIT BRANCH >>> "
        + branch
        + " <<< \n"
        + "COMMIT ID >>> "
        + cid
        + " <<<"
    )
    record_commit_info = "COMMIT TAG [\n%s]\n" % commit_tag
    return record_commit_info


def make_parser():
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
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=0, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/ai/mnt/code/YOLOX/main_yolox_base_stenosis_binary.py",
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default="/ai/mnt/code/YOLOX/yolox/models/yolox.pth",
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
        default="tensorboard",
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
    return parser


@logger.catch
def main(exp: Exp, args):
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
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    trainer = exp.get_trainer(args)
    output_exp_file = trainer.train()
    print("\n &&&&&&& eval Start &&&&&&& \n")
    os.system(
        "python {} --exp_file {}".format("/ai/mnt/code/YOLOX/eval.py", output_exp_file)
    )
    print("\n &&&&&&& eval Done &&&&&&& \n")

    print("\n @@@@@@ visualize_assign Start @@@@@@ \n")
    os.system(
        "python {} --exp_file {}".format(
            "/ai/mnt/code/YOLOX/visualize_assign.py", output_exp_file
        )
    )
    print("\n @@@@@@ visualize_assign Done @@@@@@ \n")

    print("\n ####### infer_show Start ###### \n")
    os.system(
        "python {} --exp_file {}".format(
            "/ai/mnt/code/YOLOX/infer_show.py", output_exp_file
        )
    )
    print("\n ####### infer_show Done ###### \n")


if __name__ == "__main__":
    timestamp = time.strftime("%m_%d-%H_%M", time.localtime())
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)
    args.batch_size = exp.batch_size
    args.ckpt = exp.ckpt
    exp.exp_name = timestamp + "@" + exp.exp_name
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    if args.no_debug:
        args.gitinfo = git_commit(
            work_dir="/ai/mnt/code/YOLOX", commit_info=exp.exp_name
        )
    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    setproctitle.setproctitle(exp.exp_name)

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
