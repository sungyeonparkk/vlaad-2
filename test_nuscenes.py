"""
Adapted from salesforce@LAVIS and Vision-CAIR@MiniGPT-4. Below is the original copyright:
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import video_llama.tasks as tasks
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank, init_distributed_mode
from video_llama.common.logger import setup_logger
from video_llama.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from video_llama.common.registry import registry
from video_llama.common.utils import now

# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

import yaml ## ADD


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def setup_runner(cfg):
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    init_distributed_mode(cfg.run_cfg)

    # setup_seeds(cfg.run_cfg.seed + get_rank())

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    runner_cls = get_runner_class(cfg)
    runner = runner_cls(
        cfg=cfg, job_id=job_id, task=task, model=None, datasets=datasets
    )

    return runner

args = argparse.Namespace()

# Set attributes manually
args.cfg_path = "./t.yaml"
args.options = []

cfg = Config(args)

# # Update the configuration with the loaded dictionary
# for key, value in cfg_dict.items():
#     setattr(cfg, key, value)
# # Setup the runner
# setup_seeds(cfg.run_cfg.seed + get_rank())

job_id = now()

init_distributed_mode(cfg.run_cfg)

# setup_seeds(cfg.run_cfg.seed + get_rank())

# set after init_distributed_mode() to only log on master.
setup_logger()

cfg.pretty_print()

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)

runner_cls = get_runner_class(cfg)
runner = runner_cls(
    cfg=cfg, job_id=job_id, task=task, model=None, datasets=datasets
)

sample = datasets['nuscenes_instruct']['train'].__getitem__(1)
print("Multivew img output: ", sample['multiview_image']['img'].data.shape)


# Train using the runner
# runner.train()

