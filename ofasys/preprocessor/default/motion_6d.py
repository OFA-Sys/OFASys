# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import List

import numpy as np
import torch
from transforms3d.axangles import axangle2mat

from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.module.motion_6d import (
    batch_convert_rot6d_to_bvh,
    convert_bvh_to_rot6d,
    convert_to_bvh,
    geodesic_loss,
    rectify_rot6d,
    rot6d_to_rotmat,
)
from ofasys.utils.oss import oss_get

from ..dictionary import Dictionary
from ..instruction import Slot
from .base import (
    CollateOutput,
    PreprocessConfig,
    PreprocessSkipException,
    SafeBasePreprocess,
)

logger = logging.getLogger(__name__)


@dataclass
class Motion6dPreprocessConfig(PreprocessConfig):
    # TODO(jianxin): Deduce proper rotate_order and scale_trans from a user-provided BVH header.
    rotate_order: str = field(default='XYZ', metadata={"help": ""})
    scale_trans: float = field(default=100.0, metadata={"help": ""})
    scale_rot6d: float = field(default=10.0, metadata={"help": ""})
    padding_mode: bool = field(default=True, metadata={"help": ""})
    round_to_multiple: int = field(default=1, metadata={"help": ""})


def _sample_span(n, p, min_size, max_size):
    assert min_size <= n
    assert min_size <= max_size

    if n <= min_size:
        low, high = 0, n
    else:
        while True:
            low = np.random.poisson(n * p)
            high = n - np.random.poisson(n * p)
            if high - low >= min_size:
                break

    while high - low > max_size:
        if np.random.rand() < 0.5:
            low += 1
        else:
            high -= 1
    return low, high


@register_config("ofasys.preprocess", "motion_6d", Motion6dPreprocessConfig)
class Motion6dPreprocess(SafeBasePreprocess):
    def __init__(self, global_dict: Dictionary, cfg: Motion6dPreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.MOTION)
        self.rotate_order = cfg.rotate_order
        self.scale_trans = cfg.scale_trans
        self.scale_rot6d = cfg.scale_rot6d
        self.padding_mode = cfg.padding_mode
        self.round_to_multiple = cfg.round_to_multiple
        assert self.round_to_multiple >= 1

    def _load_np(self, path):
        data = np.load(path)
        if isinstance(data, np.ndarray):
            frames = data
            weights = None
        elif isinstance(data, np.lib.npyio.NpzFile):
            frames = data['frames']
            weights = data['weights']
            data.close()  # prevent fd leaks
        else:
            raise NotImplementedError
        return {'frames': frames, 'weights': weights}

    def map(self, slot: Slot) -> Slot:
        super().map(slot)
        data = slot.value
        if isinstance(data, str):
            if data.startswith("oss://"):
                fin = oss_get(data)
                with BytesIO(fin.read()) as bio:
                    data = self._load_np(bio)
                del fin  # close
            else:
                data = self._load_np(data)
        frames, weights = data['frames'], data['weights']
        assert isinstance(frames, np.ndarray)

        sample_rate = slot.get_attr('sample_rate', int) or 1
        assert sample_rate >= 1
        if sample_rate > 1:
            frames = frames[np.random.randint(0, sample_rate) :: sample_rate]

        max_length = slot.get_attr('max_length', int)
        assert max_length, 'Please provide attribute max_length for MOTION in the instruction and ensure max_length>0.'

        if self.padding_mode:
            min_length = 1
        else:
            min_length = self.round_to_multiple

        if frames.shape[0] <= 1 + min_length:  # plus one extra frame for computing velocity
            raise PreprocessSkipException

        beg, end = _sample_span(frames.shape[0], p=0.15, min_size=min_length + 1, max_size=max_length + 1)
        frames = frames[beg:end]

        aug_rot_mat = axangle2mat([0, 1, 0], np.deg2rad(np.random.rand() * 360))  # data augmentation
        poses = convert_bvh_to_rot6d(
            frames,
            rotate_order=self.rotate_order,
            scale_trans=self.scale_trans,
            scale_rot6d=self.scale_rot6d,
            aug_rot_mat=aug_rot_mat,
        )
        assert min_length <= poses.shape[0] <= max_length

        slot.value = torch.from_numpy(poses.astype(np.float32))
        return slot

    def collate(self, slots: List[Slot]) -> CollateOutput:
        super().collate(slots)
        float_dtype, device = slots[0].value.dtype, slots[0].value.device

        lengths = [slot.value.shape[0] for slot in slots]

        input_slot = copy.copy(slots[0])
        if self.padding_mode:
            max_length = max(lengths)
            while max_length % self.round_to_multiple != 0:
                max_length += 1
            masks = torch.greater_equal(
                torch.arange(max_length, device=device).unsqueeze(0),
                torch.LongTensor(lengths, device=device).unsqueeze(-1),
            )
            value = torch.stack(
                [
                    torch.cat(
                        [
                            slot.value,
                            torch.zeros(
                                max_length - slot.value.shape[0],
                                *slot.value.shape[1:],
                                dtype=float_dtype,
                                device=device
                            ),
                        ],
                        dim=0,
                    )
                    for slot in slots
                ],
                dim=0,
            )
            assert (value.shape[:2] == masks.shape) and (len(value.shape) == 3)
            input_slot.value = {"value": value, "masks": masks}
            ntokens = sum(lengths)
        else:
            min_length = min(lengths)
            while min_length % self.round_to_multiple != 0:
                min_length -= 1
            assert min_length > 0
            value = torch.stack([slot.value[:min_length] for slot in slots], dim=0)
            input_slot.value = {"value": value}
            ntokens = value.shape[0] * value.shape[1]

        if input_slot.is_src:
            return CollateOutput(input_slot)

        target_slot = copy.copy(input_slot)
        extra_dict = {"ntokens": ntokens}
        return CollateOutput(input_slot, target_slot, extra_dict)

    def decode(self, value: torch.FloatTensor):
        if value.dim() == 2:
            value = value.unsqueeze(0)
            value = batch_convert_rot6d_to_bvh(
                value, rotate_order=self.rotate_order, scale_trans=self.scale_trans, scale_rot6d=self.scale_rot6d
            ).squeeze()
            bvh_object = convert_to_bvh(value)
        else:
            value = batch_convert_rot6d_to_bvh(
                value, rotate_order=self.rotate_order, scale_trans=self.scale_trans, scale_rot6d=self.scale_rot6d
            )
            bvh_object = [convert_to_bvh(value[i]) for i in range(value.shape[0])]
        return bvh_object

    def custom_clamp(self, value):
        assert len(value.shape) == 3
        valid_trans = value[:, :, :3]
        valid_poses = rectify_rot6d(value[:, :, 3:] / self.scale_rot6d) * self.scale_rot6d
        return torch.cat([valid_trans, valid_poses], dim=2)

    def _custom_reg_loss(self, slot: Slot, prediction, target):  # rename it to custom_reg_loss to enable it
        batch_size, seq_len, num_joints = prediction.shape
        num_joints = (num_joints - 3) // 6
        pred = rot6d_to_rotmat(prediction[:, :, 3:] / self.scale_rot6d).view(batch_size, seq_len, num_joints, 3, 3)
        target = rot6d_to_rotmat(target[:, :, 3:] / self.scale_rot6d).view(batch_size, seq_len, num_joints, 3, 3)
        loss = geodesic_loss(pred, target, reduction='none')
        if "masks" in slot.value:
            loss = loss.mean(dim=-1)  # [B,T]
            weights = torch.logical_not(slot.value["masks"]).type_as(loss)
            assert loss.shape == weights.shape
            loss = torch.mean(torch.sum(weights * loss, dim=-1) / torch.sum(weights, dim=-1), dim=-1)  # [B,T]->scalar
        else:
            loss = loss.mean()
        return loss
