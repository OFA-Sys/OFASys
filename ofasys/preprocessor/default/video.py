# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import math
import os
import random
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Union

import requests
import torch
from torchvision import transforms

from ofasys.utils.logging_utils import master_logging

_no_av_help = "No package `av` found, please install it by `pip install av` if you need to support video tasks"
logger = logging.getLogger(__name__)
try:
    import av

    _is_av_missing = 0
except ImportError as _:
    with master_logging():
        logger.info(_no_av_help)
    _is_av_missing = 1

from ofasys.configure import register_config
from ofasys.utils.oss import oss_get
from ofasys.utils.video import decoder as decoder
from ofasys.utils.video import transform as transform
from ofasys.utils.video import utils as utils
from ofasys.utils.video.random_erasing import RandomErasing
from ofasys.utils.video.transform import create_random_augment

from ..instruction import ModalityType, Slot
from ..utils import base64decode
from .base import (
    CollateOutput,
    PreprocessConfig,
    PreprocessSkipException,
    SafeBasePreprocess,
)
from .image import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)


def video_tensor_normalize(tensor, mean, std, func=None):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std)
    if func is not None:
        tensor = func(tensor)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def video_tensor_normalize_reverse(tensor, mean, std, func=None):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    assert len(mean) == len(std) and len(mean) == 3
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std)
    if func is not None:
        tensor = func(tensor)
    if tensor.size(-1) != 3:
        assert len(tensor.shape) == 4
        mean = mean.view(-1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


# TODO: Add type annotations for torchvision backend
def load_video_from_stream(stream, multi_thread_decode=False, backend="pyav"):
    # TODO: Update comments
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        container = stream.read()
        return container
    elif backend == "pyav":
        container = av.open(stream)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))


# TODO: Handle torchvision video types
def load_video(video: Union[str, 'av.container.InputContainer'], backend="pyav") -> 'av.container.InputContainer':
    # TODO: Add comment for torchvision Video
    """
    Loads `video` to a PyAV/torchvision Video.

    Args:
        video (`str` or `av.container.InputContainer`) could be:
            A remote link starts with `http://` or `https://` or `oss://`;
            A base64 string of video;
            A local file path;
            A av.container.InputContainer object

    Returns:
        `av.container.InputContainer`: A av.container.InputContainer video.
    """

    if isinstance(video, str):
        if video.startswith("http://") or video.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to
            # use a local file like http_huggingface_co.png.
            video = load_video_from_stream(BytesIO(requests.get(video, stream=True).raw.read()))
        elif video.startswith("oss://"):
            fin = oss_get(video)
            video = load_video_from_stream(BytesIO(fin.read()))
            del fin
        elif os.path.exists(video):
            video = load_video_from_stream(BytesIO(open(video, 'rb').read()))
        else:
            image_bytes = base64decode(video)
            if image_bytes is not None:
                video = load_video_from_stream(BytesIO(image_bytes))
            elif os.path.isfile(video):
                video = load_video_from_stream(open(video, 'rb').read())
            else:
                raise ValueError(f"Incorrect format used for image.{load_video.__doc__}Got {video}")
    # TODO: Handle torchvision video types
    elif isinstance(video, av.container.InputContainer):
        video = video
    else:
        raise ValueError(f"Incorrect format used for image.{load_video.__doc__}Got {video}")
    return video


@dataclass
class VideoPreprocessConfig(PreprocessConfig):
    decoding_backend: str = field(default='pyav', metadata={"help": ""})  # TODO: Add help information
    patch_image_size: int = field(default=256, metadata={"help": "The patch image size for each frame."})
    imagenet_default_mean_and_std: bool = field(
        default=True,
        metadata={"help": "imagenet normalize"},
    )
    interpolation: str = field(
        default='bicubic', metadata={"help": "Method of image interpolation used for each frame."}
    )
    train_jitter_scales_min: int = field(default=256, metadata={"help": "The min frame jittering scale."})
    train_jitter_scales_max: int = field(default=320, metadata={"help": "The max frame jittering scale."})
    train_crop_size: int = field(
        default=256,
        metadata={
            "help": "The image crop size of each frame when training."
        },  # TODO: Remove and replace with patch_image_size?
    )
    test_crop_size: int = field(
        default=256,
        metadata={
            "help": "The image crop size of each frame when testing."
        },  # TODO: Remove and replace with patch_image_size?
    )
    train_jitter_scales_relative_min: float = field(
        default=0.08, metadata={"help": "The min relative frame jittering scale."}
    )
    train_jitter_scales_relative_max: float = field(
        default=1.0, metadata={"help": "The max relative frame jittering scale."}
    )
    train_jitter_aspect_relative_min: float = field(
        default=0.75, metadata={"help": "The min frame jittering aspect ratio."}
    )
    train_jitter_aspect_relative_max: float = field(
        default=1.3333, metadata={"help": "The max frame jittering aspect ratio."}
    )
    num_frames: int = field(default=16, metadata={"help": "Number of frames to sample for each video."})
    sampling_rate: float = field(
        default=-1,
        metadata={
            "help": "The sampling rate used when sampling frames for each video. It can be positive or negative. "
            "Note that when sampling_rate > 0, it means use the sampling rate itself, i.e. samples a num_frames * sampling_rate frames clip from the video. "
            "However, when sampling_rate < 0, it means adaptively set the sampling rate to sample a -1.0 / sampling_rate portion of the whole video. "
            "i.e. when sampling_rate < 0, the actual sampling rate will approximately be video_length / (-sampling_rate) / num_frames."
        },
    )
    target_fps: int = field(default=30, metadata={"help": "The target FPS for each video."})
    train_jitter_fps: float = field(default=0.0, metadata={"help": ""})  # TODO: Add help information
    train_crop_num_spatial: int = field(default=1, metadata={"help": ""})  # TODO: Add help information
    train_aug_num_sample: int = field(default=1, metadata={"help": "The number of frames used for each frame"})

    # Configs for frame auto augment
    train_auto_augment_type: str = field(
        default="",
        metadata={
            "help": "The type of image auto augmentation for each frame, for example, rand-m7-n4-mstd0.5-inc1. When set to empty, image auto augmentation will be disabled."
        },
    )

    # Configs for frame random erase augment
    train_random_erase_prob: float = field(
        default=0.25,
        metadata={
            "help": "The probability of random erase data augmentation. When set to 0.0, random erase data augmentation will be disabled."
        },
    )
    train_random_erase_mode: str = field(
        default='pixel', metadata={"help": "The mode for random erase data augmentation."}
    )
    train_random_erase_count: int = field(
        default=1, metadata={"help": "The number of regions for random erase data augmentation."}
    )


@register_config("ofasys.preprocess", "video", VideoPreprocessConfig)
class DefaultVideoPreprocess(SafeBasePreprocess):
    def __init__(self, global_dict, cfg: VideoPreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.VIDEO)
        if cfg.imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = IMAGENET_INCEPTION_MEAN
            std = IMAGENET_INCEPTION_STD
        self.mean = list(mean)
        self.std = list(std)
        self.cfg = cfg
        self.p_convert_gray = 0.0  # self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = 0.0  # self.cfg.DATA.TIME_DIFF_PROB

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def _map(self, slot: Slot) -> Slot:
        super().map(slot)
        video = slot.value
        video_container = load_video(video)

        randaug = False
        rand_erase = False

        if slot.split == "train":  # TODO: and self.cfg.AUG.ENABLE:
            randaug = True
            if self.cfg.train_random_erase_prob > 0:
                rand_erase = True

        short_cycle_idx = None
        # TODO: The following codes are copied from SlowFast and commented as is.
        # Examine carefully to determine uncomment or delete
        # When short cycle is used, input index is a tupple.
        '''if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index'''

        if slot.split in ["train"]:  # TODO:, "valid"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.train_jitter_scales_min
            max_scale = self.cfg.train_jitter_scales_max
            crop_size = self.cfg.train_crop_size
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S)
                )
            # TODO: The following codes are copied from SlowFast and commented as is
            '''if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )'''
        elif slot.split in ["test", "valid"]:  # TODO!
            temporal_sample_index = 0
            spatial_sample_index = 1
            min_scale = self.cfg.train_jitter_scales_min
            max_scale = self.cfg.train_jitter_scales_min
            crop_size = self.cfg.test_crop_size
            # TODO: Support multi-crop and multi-clip evaluation.
            # TODO: The following codes are copied from SlowFast and commented as is
            '''temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.train_jitter_scales_min] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )'''
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(slot.split))
        # TODO: The following codes are copied from SlowFast and commented as is
        num_decode = 1
        '''num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
            if slot.split in ["train"]
            else 1
        )'''
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.train_jitter_scales_min] * (num_decode - len(min_scale))
            max_scale += [self.cfg.train_jitter_scales_max] * (num_decode - len(max_scale))
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE or self.cfg.MULTIGRID.SHORT_CYCLE
                else [self.cfg.train_crop_size] * (num_decode - len(crop_size))
            )
            assert slot.split in ["train", "valid"]
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        # for i_try in range(self._num_retries):
        if True:
            '''if True:
            #try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            print(video_container)
            exit(0)
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )'''
            # Select a random video if the current video was not able to access.
            '''if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if True: # TODO! slot.split not in ["test"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue'''

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )

            # for i in range(num_decode):
            num_frames = [self.cfg.num_frames]
            '''sampling_rate = utils.get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )'''
            sampling_rate = self.cfg.sampling_rate
            sampling_rate = [sampling_rate]
            if len(num_frames) < num_decode:
                num_frames.extend([num_frames[-1] for i in range(num_decode - len(num_frames))])
                # base case where keys have same frame-rate as query
                sampling_rate.extend([sampling_rate[-1] for i in range(num_decode - len(sampling_rate))])
            elif len(num_frames) > num_decode:
                num_frames = num_frames[:num_decode]
                sampling_rate = sampling_rate[:num_decode]

            if slot.split in ["train"]:
                assert len(min_scale) == len(max_scale) == len(crop_size) == num_decode

            target_fps = self.cfg.target_fps
            if self.cfg.train_jitter_fps > 0.0 and slot.split in ["train"]:
                target_fps += random.uniform(0.0, self.cfg.train_jitter_fps)

            # Decode video. Meta info is used to perform selective decoding.
            frames, time_idx, tdiff = decoder.decode(
                video_container,
                sampling_rate,
                num_frames,
                temporal_sample_index,
                1,  # TODO: self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta={},  # self._video_meta[index] if len(self._video_meta) < 5e6 else {},  # do not cache on huge datasets
                target_fps=target_fps,
                backend=self.cfg.decoding_backend,
                use_offset=True,  # self.cfg.USE_OFFSET_SAMPLING,
                max_spatial_scale=min_scale[0]
                if all(x == min_scale[0] for x in min_scale)
                else 0,  # if slot.split in ["test"] else 0,
                time_diff_prob=self.p_convert_dt if slot.split in ["train"] else 0.0,
                temporally_rnd_clips=True,
                min_delta=-math.inf,  # TODO: self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=math.inf,  # TODO: self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )

            frames_decoded = frames
            time_idx_decoded = time_idx

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            '''if frames_decoded is None or None in frames_decoded:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if (
                    slot.split not in ["test"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                # TODO!
                if True:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue'''

            num_aug: int = (
                self.cfg.train_crop_num_spatial * self.cfg.train_aug_num_sample if slot.split in ["train"] else 1
            )
            num_out = num_aug * num_decode
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            # label = self._labels[index]

            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    f_out[idx] = frames_decoded[i].clone()
                    time_idx_out[idx] = time_idx_decoded[i, :]

                    f_out[idx] = f_out[idx].float()
                    f_out[idx] = f_out[idx] / 255.0

                    # TODO
                    '''if (
                        slot.split in ["train"]
                        and self.cfg.DATA.SSL_COLOR_JITTER
                    ):
                        f_out[idx] = transform.color_jitter_video_ssl(
                            f_out[idx],
                            bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                            hue=self.cfg.DATA.SSL_COLOR_HUE,
                            p_convert_gray=self.p_convert_gray,
                            moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                            gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                            gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                        )'''

                    if slot.split == 'train' and self.cfg.train_auto_augment_type.strip():
                        aug_transform = create_random_augment(
                            input_size=(f_out[idx].size(1), f_out[idx].size(2)),
                            auto_augment=self.cfg.train_auto_augment_type.strip(),
                            interpolation=self.cfg.interpolation,
                        )
                        # T H W C -> T C H W.
                        f_out[idx] = f_out[idx].permute(0, 3, 1, 2)
                        list_img = self._frame_to_list_img(f_out[idx])
                        list_img = aug_transform(list_img)
                        f_out[idx] = self._list_img_to_frames(list_img)
                        f_out[idx] = f_out[idx].permute(0, 2, 3, 1)

                    # Perform color normalization.
                    f_out[idx] = utils.tensor_normalize(f_out[idx], self.mean, self.std)

                    # T H W C -> C T H W.
                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                    scl, asp = (
                        [
                            self.cfg.train_jitter_scales_relative_min,
                            self.cfg.train_jitter_scales_relative_max,
                        ],
                        [
                            self.cfg.train_jitter_aspect_relative_min,
                            self.cfg.train_jitter_aspect_relative_max,
                        ],
                    )
                    relative_scales = None if (slot.split not in ["train"] or len(scl) == 0) else scl
                    relative_aspect = None if (slot.split not in ["train"] or len(asp) == 0) else asp
                    f_out[idx] = utils.spatial_sampling(
                        f_out[idx],
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale[i],
                        max_scale=max_scale[i],
                        crop_size=crop_size[i],
                        random_horizontal_flip=True,  # TODO: self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=False,  # TODO: self.cfg.DATA.INV_UNIFORM_SAMPLE,
                        aspect_ratio=relative_aspect,
                        scale=relative_scales,
                        motion_shift=False
                        if slot.split in ["train"]
                        else False,  # TODO: self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                    )

                    if rand_erase:
                        erase_transform = RandomErasing(
                            self.cfg.train_random_erase_prob,
                            mode=self.cfg.train_random_erase_mode,
                            max_count=self.cfg.train_random_erase_count,
                            num_splits=self.cfg.train_random_erase_count,
                            device="cpu",
                        )
                        f_out[idx] = erase_transform(f_out[idx].permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

                    f_out[idx] = [f_out[idx]]  # TODO: utils.pack_pathway_output(self.cfg, f_out[idx])
            frames = f_out[0] if num_out == 1 else f_out
            # TODO: Currently we assume that there would only be one video as the output of the preprocessor.
            assert len(frames) == 1
            slot.value = frames[0]
            return slot

    def map(self, slot: Slot) -> Slot:
        if isinstance(slot.value, torch.Tensor):
            return slot
        try:
            return self._map(slot=slot)
        except Exception as e:
            print(e)
            raise PreprocessSkipException()

    def collate(self, slots: List[Slot]) -> CollateOutput:
        super().collate(slots)
        slots[0].value = torch.stack([slot.value for slot in slots], dim=0)
        slot = slots[0]
        return CollateOutput(slot)
