#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger
import glob
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger,
)


def write_to_csv(filename, content):
    file_exist = os.path.exists(filename)
    with open(filename, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        if not file_exist:
            writer.writeheader()
        writer.writerow(content)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

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
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/ai/mnt/code/YOLOX/output_runs/Binary/06_28-16_28@YOLOX-ALL-NMS0.35-V1024-SR8-CROP0.5_first_256-ZOOM0.35-MOTION.04-tf5e_3-cutcopy_ex10/main_yolox_base_stenosis_binary_copy.py",
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str,
        help="ckpt for eval",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument(
        "--box_contain_thresh", default=None, type=float, help="box_contain_thresh"
    )
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.dirname(args.exp_file)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.box_contain_thresh is not None:
        exp.box_contain_thresh = args.box_contain_thresh
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    if args.ckpt is None:
        args.ckpt = glob.glob(
            os.path.join(os.path.dirname(args.exp_file), "*best*.pth")
        )[0]
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(
        args.batch_size, is_distributed, args.test, args.legacy
    )
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    results, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    csvname = "/ai/mnt/code/YOLOX/Binay_AP.csv"
    csv_content = dict(
        ExpName=exp.exp_name,
        AP5095=results[0],
        AP50=results[1],
        AP75=results[2],
        AP5095Small=results[3],
        AP5095Medium=results[4],
        AP5095Large=results[5],
        AR5095All=results[8],
        AR5095Small=results[9],
        AR5095Medium=results[10],
        AR5095Large=results[11],
        Box_Contain_Thresh=exp.box_contain_thresh,
    )
    write_to_csv(csvname, csv_content)
    logger.info("\n" + summary)
    logger.info("Write {} to csv to {}".format(csv_content, csvname))
    return results


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
    print("\n @@@@@@ visualize_assign Start @@@@@@ \n")
    os.system(
        "python {} --exp_file {}".format(
            "/ai/mnt/code/YOLOX/visualize_assign.py", args.exp_file
        )
    )
    print("\n @@@@@@ visualize_assign Done @@@@@@ \n")
    print("\n ####### infer_show Start ###### \n")
    os.system(
        "python {} --exp_file {}".format(
            "/ai/mnt/code/YOLOX/infer_show.py", args.exp_file
        )
    )
    print("\n ####### infer_show Done ###### \n")
