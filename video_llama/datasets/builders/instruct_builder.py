import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.laion_dataset import LaionDataset

from video_llama.datasets.datasets.llava_instruct_dataset import Instruct_Dataset
from video_llama.datasets.datasets.video_instruct_dataset import Video_Instruct_Dataset
from video_llama.datasets.datasets.bdd_instruct_dataset import BDD_Instruct_Dataset
from video_llama.datasets.datasets.had_instruct_dataset import HAD_Instruct_Dataset
from video_llama.datasets.datasets.maplm_instruct_dataset import MAPLM_Instruct_Dataset
from video_llama.datasets.datasets.drama_instruct_dataset import DRAMA_Instruct_Dataset
from video_llama.datasets.datasets.nuscenes_instruct_dataset import NUSCENES_Instruct_Dataset

@registry.register_builder("instruct")
class Instruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Instruct_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/instruct/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()

        for split in ["train"]:
            build_info = self.config.build_info[split]
            dataset_cls = self.train_dataset_cls
            if self.config.num_video_query_token:
                num_video_query_token = self.config.num_video_query_token
            else:
                num_video_query_token = 32

            if self.config.tokenizer_name:
                tokenizer_name = self.config.tokenizer_name
            else:
                tokenizer_name = "/mnt/workspace/ckpt/vicuna-13b/"

            datasets[split] = dataset_cls(
                vis_processor=self.vis_processors[split],
                text_processor=self.text_processors[split],
                vis_root=build_info.videos_dir,
                ann_root=build_info.anno_dir,
                num_video_query_token=num_video_query_token,
                tokenizer_name=tokenizer_name,
                data_type=self.config.data_type,
            )

        return datasets


@registry.register_builder("webvid_instruct")
class WebvidInstruct_Builder(Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }


@registry.register_builder("webvid_instruct_zh")
class WebvidInstruct_zh_Builder(Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }


@registry.register_builder("llava_instruct")
class LlavaInstruct_Builder(Instruct_Builder):
    train_dataset_cls = Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/llava_instruct.yaml",
    }


@registry.register_builder("bdd_instruct")
class BDDInstruct_Builder(Instruct_Builder):
    train_dataset_cls = BDD_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/bdd_instruct.yaml",
    }


@registry.register_builder("had_instruct")
class HADInstruct_Builder(Instruct_Builder):
    train_dataset_cls = HAD_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/had_instruct.yaml",
    }


@registry.register_builder("maplm_instruct")
class MAPLMInstruct_Builder(Instruct_Builder):
    train_dataset_cls = MAPLM_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/maplm_instruct.yaml",
    }


@registry.register_builder("drama_instruct")
class DRAMAInstruct_Builder(Instruct_Builder):
    train_dataset_cls = DRAMA_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/drama_instruct.yaml",
    }


@registry.register_builder("nuscenes_instruct")
class NUSCENESInstruct_Builder(Instruct_Builder):
    train_dataset_cls = NUSCENES_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/nuscenes_instruct.yaml",
    }
