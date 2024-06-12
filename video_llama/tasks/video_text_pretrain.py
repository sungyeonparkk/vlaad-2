"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from video_llama.common.registry import registry
from video_llama.tasks.base_task import BaseTask

import torch.distributed as dist
from video_llama.common.logger import MetricLogger, SmoothedValue
from video_llama.datasets.data_utils import prepare_sample
from video_llama.common.dist_utils import (
    get_rank,
    get_world_size,
    is_main_process,
    is_dist_avail_and_initialized,
)


@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 1

        results = []
        
        iter_count = 0
        for data in data_loader:
            print(f"VALID STEP COUNT : {iter_count}")
            if data == None:
               continue
            samples = prepare_sample(data, cuda_enabled=cuda_enabled)
            eval_output, metric_dict = self.valid_step(model=model, samples=samples)
            # print("Input Ids: ", samples["input_ids"][0])
            # print("Labels: ", samples["labels"][0])
            try:
                results.extend(eval_output)
            except TypeError:
                results.append(eval_output)

            iter_count += 1
            if iter_count == 15:
               break

        # Presumed to be code for DDP. But code output error when it is uncommented. Refer to notion page.
        # if is_dist_avail_and_initialized():
        #     dist.barrier()

        return results, metric_dict
