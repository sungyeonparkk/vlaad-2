import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import numpy as np
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
from video_llama.processors import transforms_video, AlproVideoTrainProcessor
from torchvision import transforms
import mmcv
from mmcv.parallel import DataContainer as DC
from collections import defaultdict, OrderedDict

from video_llama.processors.video_processor import ToTHWC, ToUint8, load_video
from video_llama.conversation.conversation_video import Conversation, SeparatorStyle
from video_llama.processors.process_multivew import MultiviewImageProcessingPipeline

from mmdet.datasets.pipelines import to_tensor
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


DEFAULT_IMAGE_PATCH_TOKEN = "<ImageHere>"
video_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
llama_v2_video_conversation = Conversation(
    system=" ",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
IGNORE_INDEX = -100


class NUSCENES_Instruct_Dataset(BaseDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_root,
        num_video_query_token=32,
        tokenizer_name="/input/video-llama-ckpt-2/ckpt/llama-2-7b-chat-hf",
        data_type="video",
        model_type="llama_v2",
    ):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)

        with data_path.open(encoding="utf-8") as f:
            self.annotation = json.load(f)

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = vis_processor.image_size
        self.num_frm = vis_processor.n_frms
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name, use_fast=False, local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[
            DEFAULT_IMAGE_PATCH_TOKEN
        ]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms=self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type

        self.file_client_args = dict(backend='disk')
        self.normalization_params = {
            'mean': [103.530, 116.280, 123.675],
            'std': [1.0, 1.0, 1.0],
            'to_rgb': False
        }
        self.pad_params = {
            'size': None,
            'size_divisor': 32,
            'pad_val': 0
        }
        self.class_names = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]
        self.keys = ["img", "timestamp", "lidar2ego_rotation", "lidar2ego_translation", "ego2global_rotation", "ego2global_translation"]
        self.img_root = "/input/vision-assistant-for-driving/data/NUSCENES_train_data/nuscenes/nuscenes/"

        ann_file_root = "/input/vision-assistant-for-driving/data/NUSCENES_train_data/nuscenes/nuscenes_infos_temporal_train.pkl"
        self.data_infos = mmcv.load(ann_file_root)
        self.modality = dict(
            use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
        )
        self.queue_length = 5

    def _get_video_path(self, sample):
        rel_video_fp = sample["video_id"]
        full_video_fp = os.path.join(self.vis_root, rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                if isinstance(index, str):
                   raise TypeError(f"String Index Passed!: {index}")
                sample = self.annotation[index]
                video_path = self._get_video_path(sample)
                conversation_list = sample["QA"]
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling="uniform",
                    return_msg=True,
                )
                video = self.transform(video)
                if "cn" in self.data_type:
                    msg = ""
                # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0
                sources = preprocess_multimodal(
                    copy.deepcopy(conversation_list),
                    None,
                    cur_token_len=self.num_video_query_token,
                    msg=msg,
                )
                new_sources = convert_source_vicuna_format(sources)

                if self.model_type == "vicuna":
                    data_dict = preprocess(new_sources, self.tokenizer)
                elif self.model_type == "llama_v2":
                    data_dict = preprocess_for_llama_v2(new_sources, self.tokenizer)
                else:
                    print("not support")
                    raise ("not support")
                data_dict = dict(
                    input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
                )
                # image exist in the data
                data_dict["image"] = video
           
                # Processing multi-view image processing
                multiview_img = self.prepare_nuscenes_data(sample["sample_token"], self.queue_length)
               
            except TypeError as e:
                print(e)
                index = random.randint(0, len(self) - 1)
                continue

            except Exception as e:
                print(
                    f"Failed to load examples with video: {video_path}. "
                    f"Will randomly sample an example as a replacement."
                )
                print(e)
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "multiview_image": multiview_img,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type": "video",
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("text_input", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
        batch["conv_type"] = "multi"
        return batch

    def get_data_info(self, sample_token):
        info = self.data_infos[sample_token]
        input_dict = self.prepare_input_dict(info)
        """
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        if not self.test_mode and self.mono_cfg is not None:
            if input_dict is None:
                return None
            info = self.data_infos[index]
            img_ids = []
            for cam_type, cam_info in info['cams'].items():
                img_ids.append(cam_info['sample_data_token'])

            mono_input_dict = []; mono_ann_index = []
            for i, img_id in enumerate(img_ids):
                tmp_dict = self.mono_dataset.getitem_by_datumtoken(img_id)
                if tmp_dict is not None:
                    if self.filter_crowd_annotations(tmp_dict):
                        mono_input_dict.append(tmp_dict)
                        mono_ann_index.append(i)

            # filter empth annotation
            if len(mono_ann_index) == 0:
                return None

            mono_ann_index = DC(mono_ann_index, cpu_only=True)
            input_dict['mono_input_dict'] = mono_input_dict
            input_dict['mono_ann_idx'] = mono_ann_index
        """
        return input_dict

    def prepare_input_dict(self, info):
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            prev=info['prev'],
            next=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        return input_dict


    def union2one(self, queue: dict):
        # convert sample queue into one single sample.
        imgs_list = [each['img'].data for each in queue]
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(queue[0]['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = queue[0]['lidar2ego_translation']

        egocurr2global = np.eye(4, dtype=np.float32)
        egocurr2global[:3,:3] = Quaternion(queue[0]['ego2global_rotation']).rotation_matrix
        egocurr2global[:3,3] = queue[0]['ego2global_translation']
        metas_map = {}
        for i in range(len(queue)):
            each = queue[i]
            metas_map[i] = each['img_metas'].data
            metas_map[i]['timestamp'] = each['timestamp']
            if 'aug_param' in each:
                metas_map[i]['aug_param'] = each['aug_param']
            if i == 0:
                metas_map[i]['lidaradj2lidarcurr'] = None
            else:
                egoadj2global = np.eye(4, dtype=np.float32)
                egoadj2global[:3,:3] = Quaternion(each['ego2global_rotation']).rotation_matrix
                egoadj2global[:3,3] = each['ego2global_translation']

                lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) @ egoadj2global @ lidar2ego
                metas_map[i]['lidaradj2lidarcurr'] = lidaradj2lidarcurr
                for i_cam in range(len(metas_map[i]['lidar2img'])):
                    metas_map[i]['lidar2img'][i_cam] = metas_map[i]['lidar2img'][i_cam] @ np.linalg.inv(lidaradj2lidarcurr)
        queue[0]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[0]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[0]
        return queue

    """
    def union2one(self, queue):
        # convert sample dict into one single sample.
        imgs_list = [each['img'].data for each in queue]
        print(queue[0]['img_metas'].data)
        #gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        #gt_sdc_label_list = [each['gt_sdc_label'].data for each in queue]
        #gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]
        #gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        #gt_past_traj_list = [to_tensor(each['gt_past_traj']) for each in queue]
        #gt_past_traj_mask_list = [
        #    to_tensor(each['gt_past_traj_mask']) for each in queue]
        #gt_sdc_bbox_list = [each['gt_sdc_bbox'].data for each in queue]
        #l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]
        #l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]
        #timestamp_list = [to_tensor(each['timestamp']) for each in queue]
        #gt_fut_traj = to_tensor(queue[-1]['gt_fut_traj'])
        #gt_fut_traj_mask = to_tensor(queue[-1]['gt_fut_traj_mask'])
        #gt_sdc_fut_traj = to_tensor(queue[-1]['gt_sdc_fut_traj'])
        #gt_sdc_fut_traj_mask = to_tensor(queue[-1]['gt_sdc_fut_traj_mask'])
        #gt_future_boxes_list = queue[-1]['gt_future_boxes']
        #gt_future_labels_list = [to_tensor(each)
        #                            for each in queue[-1]['gt_future_labels']]

        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                                cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        #queue['gt_labels_3d'] = DC(gt_labels_3d_list)
        #queue['gt_sdc_label'] = DC(gt_sdc_label_list)
        #queue['gt_inds'] = DC(gt_inds_list)
        #queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
        #queue['gt_sdc_bbox'] = DC(gt_sdc_bbox_list, cpu_only=True)
        #queue['l2g_r_mat'] = DC(l2g_r_mat_list)
        #queue['l2g_t'] = DC(l2g_t_list)
        #queue['timestamp'] = DC(timestamp_list)
        #queue['gt_fut_traj'] = DC(gt_fut_traj)
        #queue['gt_fut_traj_mask'] = DC(gt_fut_traj_mask)
        #queue['gt_past_traj'] = DC(gt_past_traj_list)
        #queue['gt_past_traj_mask'] = DC(gt_past_traj_mask_list)
        #queue['gt_future_boxes'] = DC(gt_future_boxes_list, cpu_only=True)
        #queue['gt_future_labels'] = DC(gt_future_labels_list)
        return queue
    """

    def prepare_nuscenes_data(self, sample_token, queue_length):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        data_queue = []
        input_dict = self.get_data_info(sample_token)
        cur_scene_token = input_dict['scene_token']

        pipeline = MultiviewImageProcessingPipeline(self.file_client_args, self.img_root, self.normalization_params, self.pad_params, self.keys, self.class_names)
        self.pre_pipeline(input_dict)
        example = pipeline.process(input_dict)
        data_queue.insert(0, copy.deepcopy(example))
        
        for frame_idx in range(queue_length):
            chosen_idx = input_dict['prev']
            if frame_idx == 0 or len(chosen_idx) == 0:
                continue
            info = self.data_infos[chosen_idx]
            input_dict = self.prepare_input_dict(info)
            if input_dict['scene_token'] == cur_scene_token:
                self.pre_pipeline(input_dict)
                example = pipeline.process(input_dict)
            data_queue.insert(0, copy.deepcopy(example))
        data_queue = self.union2one(data_queue)
        return data_queue


    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        """
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        """

def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence["q"]
            role_1_msg = sentence["a"]
            new_source.append(
                {
                    "from": "human",
                    "value": role_0_msg,
                }
            )
            new_source.append(
                {
                    "from": "gpt",
                    "value": role_1_msg,
                }
            )
        new_sources.append(new_source)
    return new_sources


def preprocess_multimodal(
    conversation_list: Sequence[str], multimodal_cfg: dict, cur_token_len: int, msg=""
) -> Dict:
    # 将conversational list中
    is_multimodal = True
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    conversation_list[0]["q"] = (
        "<Video>"
        + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        + "</Video> "
        + msg
        + conversation_list[0]["q"]
    )
    return [conversation_list]


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_for_llama_v2(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn(
            [header] + [s["value"] for s in source], tokenizer
        )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    conv = copy.deepcopy(video_conversation.copy())
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for source in sources:
        # <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n
        header = f"<s>[INST] <<SYS>>\n{conv.system}\n</SYS>>\n\n"

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=512,
        truncation=True,
    ).input_ids
    targets = copy.deepcopy(input_ids)

    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = (
                len(tokenizer(parts[0]).input_ids) - 2
            )  # 为什么减去2,speical token 的数目

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len



