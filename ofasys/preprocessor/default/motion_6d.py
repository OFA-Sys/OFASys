# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F

from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.module.motion_6d import (
    BvhHeader,
    rectify_rot6d,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
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
    bvh_header: str = field(
        default="oss://ofasys/data/human_motion/smplh_bvh_header.bvh",
        metadata={
            "help": "SMPL-H: oss://ofasys/data/human_motion/smplh_bvh_header.bvh, "
            "LaFAN: oss://ofasys/data/human_motion/lafan_train/lafan_bvh_header.bvh"
        },
    )
    inbetween_args: str = field(
        default="", metadata={"help": 'e.g., {"n_past": [10,11], "n_miss": [5, 40], "n_next": [1,2]}'}
    )


def sample_center_span(n, min_size, max_size, p):
    assert min_size <= n
    assert min_size <= max_size

    if n <= min_size:
        low, high = 0, n
    else:
        while True:
            low = np.random.poisson(n * p * 0.5)
            high = n - np.random.poisson(n * p * 0.5)
            if high - low >= min_size:
                break

    while high - low > max_size:
        if np.random.rand() < 0.5:
            low += 1
        else:
            high -= 1
    return low, high


def sample_uniform_span(n, min_size, max_size):
    sz = np.random.randint(min_size, max_size + 1)
    sz = min(sz, n)
    low = np.random.randint(0, n - sz + 1)
    high = low + sz
    return low, high


@register_config("ofasys.preprocess", "motion_6d", Motion6dPreprocessConfig)
class Motion6dPreprocess(SafeBasePreprocess):
    def __init__(self, global_dict: Dictionary, cfg: Motion6dPreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.MOTION)
        assert len(cfg.bvh_header) > 0, 'Please provide the path to the BVH header.'
        self.bvh_header = BvhHeader(path=cfg.bvh_header)
        self.rotate_order = self.bvh_header.get_rotation_order()
        if len(cfg.inbetween_args) > 0:
            self.inbetween_args = json.loads(cfg.inbetween_args)
        else:
            self.inbetween_args = None
        self.scale_velocity = 100.0
        self._oss_cache = {}

    def _load_data_from_file(self, path):
        def _load_np(file):
            d = np.load(file)
            if isinstance(d, np.ndarray):
                frames = d
                known_w = None
            elif isinstance(d, np.lib.npyio.NpzFile):
                frames = d['frames']
                known_w = d['weights']
                assert known_w.shape == frames.shape[:-1]  # frame-level masks for motion in-betweening
                d.close()  # prevent fd leaks
            else:
                raise NotImplementedError
            return {'frames': frames, 'known_w': known_w}

        if path.startswith("oss://"):
            if path in self._oss_cache:
                data = self._oss_cache[path]
            else:
                fin = oss_get(path)
                with BytesIO(fin.read()) as bio:
                    data = _load_np(bio)
                del fin  # close
                self._oss_cache[path] = data
        else:
            data = _load_np(path)
        return data

    def _convert_bvh_to_rot6d(
        self, frames: np.ndarray, orient_rotate: Optional[np.ndarray] = None, trans_offset: Optional[np.ndarray] = None
    ) -> np.ndarray:
        num_frames, num_joints = frames.shape
        num_joints = (num_joints - 3) // 3

        rot_mats = R.from_euler(self.rotate_order, frames[:, 3:].reshape((num_frames * num_joints, 3)), degrees=True)
        rot_mats = rot_mats.as_matrix().reshape((num_frames, num_joints, 3, 3))
        trans = frames[:, :3].copy()

        if trans_offset is not None:
            assert trans_offset.shape == (3,)
            trans += trans_offset[None, :]
        if orient_rotate is not None:
            assert orient_rotate.shape == (3, 3)
            rot_mats[:, 0] = np.matmul(orient_rotate[None, :, :], rot_mats[:, 0])
            trans = np.matmul(orient_rotate[None, :, :], trans[:, :, None]).squeeze(-1)

        velocity = np.concatenate([np.zeros((1, 3), dtype=trans.dtype), trans[1:] - trans[:-1]], axis=0)
        velocity *= self.scale_velocity
        inv_root_orient = -R.from_matrix(rot_mats[:, 0]).as_euler('xzy')[:, -1]
        inv_root_orient = R.from_euler('y', inv_root_orient).as_matrix()
        velocity = np.matmul(inv_root_orient, velocity[:, :, None]).squeeze(-1)

        pose_seq = np.concatenate(
            [trans, velocity, rotmat_to_rot6d(rot_mats).reshape((num_frames, num_joints * 6))], axis=-1
        )
        assert pose_seq.shape == (num_frames, 6 + 6 * num_joints)
        return pose_seq

    def _batch_convert_rot6d_to_bvh(
        self,
        trans_and_poses: np.ndarray,
        orient_rotate: Optional[np.ndarray] = None,
        trans_offset: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        batch_size, seq_len, embed_dim = trans_and_poses.shape
        num_joints = (embed_dim - 6) // 6

        trans = trans_and_poses[:, :, :3]  # [B,T,3]
        velocity = trans_and_poses[:, :, 3:6]  # [B,T,3]
        rot_mats = rot6d_to_rotmat(
            trans_and_poses[:, :, 6:].reshape(batch_size, seq_len, num_joints, 6)
        )  # [B,T,J,3,3]

        if orient_rotate is not None:
            assert orient_rotate.shape == (batch_size, 3, 3)
            inv_orient_rotate = orient_rotate.transpose((0, 2, 1))[:, None].repeat(seq_len, axis=1)  # [B,T,3,3]
            rot_mats[:, :, 0] = np.matmul(inv_orient_rotate, rot_mats[:, :, 0])  # [B,T,3,3]x[B,T,3,3]
            trans = np.matmul(inv_orient_rotate, trans[:, :, :, None]).squeeze(-1)

        if trans_offset is not None:
            assert trans_offset.shape == (batch_size, 3)
            trans = trans - trans_offset[:, None]  # [B,T,3]+[B,1,3]

        if self.inbetween_args is None:  # if not in-between
            root_orient = R.from_matrix(rot_mats[:, :, 0].reshape(batch_size * seq_len, 3, 3)).as_euler('xzy')[:, -1]
            root_orient = R.from_euler('y', root_orient).as_matrix().reshape((batch_size, seq_len, 3, 3))
            velocity = np.matmul(root_orient, velocity[:, :, :, None]).squeeze(-1)  # [B,T,3,3]x[B,T,3,1]
            velocity /= self.scale_velocity
            cum_delta_trans = np.cumsum(velocity, axis=1)
            trans[..., 0] = cum_delta_trans[..., 0] + trans[..., :1, 0]
            trans[..., 2] = cum_delta_trans[..., 2] + trans[..., :1, 2]

        eulers = R.from_matrix(rot_mats.reshape(batch_size * seq_len * num_joints, 3, 3))
        eulers = eulers.as_euler(self.rotate_order, degrees=True).reshape((batch_size, seq_len, num_joints * 3))

        bvh_frames = np.concatenate([trans, eulers], axis=-1)
        assert bvh_frames.shape == (batch_size, seq_len, 3 + 3 * num_joints)
        return bvh_frames

    def get_data_dim(self) -> int:
        # 3 (hip position), 3 (hip velocity), 6 * #joints (all rotations)
        return 6 + 6 * self.bvh_header.n_inner_joints

    def map(self, slot: Slot) -> Slot:
        super().map(slot)

        min_length = slot.get_attr('min_length', int) or 10
        max_length = slot.get_attr('max_length', int) or 1000

        data = slot.value
        if data is None:
            assert slot.split == 'test'
            slot.value = {'dummy': min(max(np.random.poisson(max_length / 8), min_length), max_length)}
            return slot

        if isinstance(data, str):
            data = self._load_data_from_file(data)
        frames, known_w = data['frames'], data['known_w']
        assert isinstance(frames, np.ndarray)

        sample_rate = slot.get_attr('sample_rate', int) or 1
        assert sample_rate >= 1
        if sample_rate > 1:
            beg = np.random.randint(0, sample_rate)
            frames = frames[beg::sample_rate]
            if known_w is not None:
                known_w = known_w[beg::sample_rate]

        if (known_w is None) and (self.inbetween_args is not None):
            n_past, n_miss, n_next = self._sample_inbetween_region()
            assert min_length <= n_past + n_miss + n_next <= max_length
            min_length = max_length = n_past + n_miss + n_next
        else:
            n_past = n_miss = n_next = 0

        if frames.shape[0] < min_length:
            raise PreprocessSkipException

        soft_trim = slot.get_attr('soft_trim', float)
        if soft_trim is None:
            beg, end = 0, frames.shape[0]
        else:
            assert 0 < soft_trim < 1.0
            beg, end = sample_center_span(frames.shape[0], min_size=min_length, max_size=frames.shape[0], p=soft_trim)
        if slot.has_attr('window'):
            delta_beg, delta_end = sample_uniform_span(end - beg, min_size=min_length, max_size=max_length)
            assert beg <= beg + delta_beg < beg + delta_end <= end
            beg, end = beg + delta_beg, beg + delta_end
        else:
            end = min(end, beg + max_length)

        frames = frames[beg:end]
        if known_w is not None:
            known_w = known_w[beg:end]

        sample = {}

        anchor_frame = slot.get_attr('anchor_frame', int)
        if anchor_frame is None:
            orient_rotate = trans_offset = None
        else:
            assert 0 <= anchor_frame < min_length
            assert (known_w is None) or (known_w[anchor_frame] > 0.5)
            orient_rotate = frames[anchor_frame, 3:6]
            orient_rotate = R.from_euler(self.rotate_order, orient_rotate, degrees=True).as_matrix()
            orient_rotate = -R.from_matrix(orient_rotate).as_euler('xzy', degrees=True)[-1]  # extrinsic y
            orient_rotate = R.from_euler('y', orient_rotate, degrees=True).as_matrix()
            trans_offset = -frames[anchor_frame, :3].copy()
            for i, a in enumerate(self.rotate_order):
                if a == 'Y':
                    trans_offset[i] = 0.0
            if self.inbetween_args is not None:
                sample['rotate_y'] = orient_rotate.astype(np.float32)
                sample['offset_xz'] = trans_offset.astype(np.float32)

        frames = self._convert_bvh_to_rot6d(frames, orient_rotate=orient_rotate, trans_offset=trans_offset)
        assert min_length <= frames.shape[0] <= max_length
        frames = torch.from_numpy(frames.astype(np.float32))
        sample['value'] = frames

        if n_miss > 0:
            assert known_w is None
            assert frames.shape[:-1] == (n_past + n_miss + n_next,)
            known_w = np.ones_like(frames[..., 0])
            known_w[n_past : (n_past + n_miss)] = 0.0
        if known_w is not None:
            known_w = torch.from_numpy(known_w.astype(np.float32))
            assert known_w.shape == frames.shape[:-1]
            sample['known_w'] = known_w
            sample['interp_w'] = self._get_interpolate_weight(known_w)

        slot.value = sample
        return slot

    def _sample_inbetween_region(self):
        n_past = np.random.randint(*self.inbetween_args['n_past'])
        if self.inbetween_args.get('bias_short', False):
            choice_values = np.arange(*self.inbetween_args['n_miss'])
            choice_weights = 1.0 / choice_values
            choice_weights = choice_weights / choice_weights.sum()
            n_miss = np.random.choice(choice_values, replace=False, p=choice_weights)
        else:
            n_miss = np.random.randint(*self.inbetween_args['n_miss'])
        n_next = np.random.randint(*self.inbetween_args['n_next'])
        return n_past, n_miss, n_next

    @staticmethod
    def _get_interpolate_weight(known_w):
        (n,) = known_w.shape
        last_k: List[Optional[int]] = [None] * n
        for k in range(n):
            if known_w[k] > 0.5:
                last_k[k] = k
            elif k - 1 >= 0:
                last_k[k] = last_k[k - 1]
        next_k: List[Optional[int]] = [None] * n
        for k in range(n - 1, -1, -1):
            if known_w[k] > 0.5:
                next_k[k] = k
            elif k + 1 < n:
                next_k[k] = next_k[k + 1]
        interp_w = torch.zeros(n, n, dtype=torch.float32)
        for k in range(n):
            i, j = last_k[k], next_k[k]
            if known_w[k] > 0.5:
                interp_w[k, k] = 1.0
            elif i is None:
                assert k < j < n
                interp_w[k, j] = 1.0
            elif j is None:
                assert 0 <= i < k
                interp_w[k, i] = 1.0
            else:
                assert 0 <= i < k < j < n
                c = (k - i) / (j - i)
                interp_w[k, i] = 1.0 - c
                interp_w[k, j] = c
        return interp_w

    def collate(self, slots: List[Slot]) -> CollateOutput:
        super().collate(slots)

        batch_size = len(slots)

        if 'dummy' in slots[0].value:
            num_frames = np.random.choice([slot.value['dummy'] for slot in slots])
            data_dim = self.get_data_dim()
            for slot in slots:
                slot.value = {'value': torch.zeros(size=(num_frames, data_dim), dtype=torch.float32)}

        lengths = [slot.value['value'].shape[0] for slot in slots]
        ntokens = sum(lengths)
        max_length = max(lengths)
        masks = torch.greater_equal(torch.arange(max_length).unsqueeze(0), torch.LongTensor(lengths).unsqueeze(-1))

        def _pad_and_collate(k):
            v_list = []
            for slot in slots:
                v = slot.value[k]
                v = torch.cat([v, torch.zeros(max_length - v.shape[0], *v.shape[1:], dtype=v.dtype)], dim=0)
                v_list.append(v)
            collated_v = torch.stack(v_list, dim=0)
            assert collated_v.shape[:2] == masks.shape
            return collated_v

        input_slot = copy.copy(slots[0])
        input_slot.value = {'value': _pad_and_collate('value'), 'masks': masks}
        for optional_k in [
            'known_w',
        ]:
            if optional_k in slots[0].value:
                input_slot.value[optional_k] = _pad_and_collate(optional_k)
        for optional_k in [
            'rotate_y',
            'offset_xz',
        ]:
            if optional_k in slots[0].value:
                # noinspection PyTypeChecker
                input_slot.value[optional_k] = np.stack([s.value[optional_k] for s in slots], axis=0)

        if 'interp_w' in slots[0].value:
            interp_w = torch.zeros(batch_size, max_length, max_length)
            for i, s in enumerate(slots):
                w = s.value['interp_w']
                n, m = w.shape
                assert n == m
                interp_w[i, :n, :m] = w
            input_slot.value['interp_w'] = interp_w

        # Keep a clean copy of 'value' as 'value_0', as diffusion will add noise to 'value'.
        input_slot.value['value_0'] = input_slot.value['value']

        if input_slot.is_src:
            return CollateOutput(input_slot)
        extra_dict = {"ntokens": ntokens}
        return CollateOutput(input_slot, None, extra_dict)

    @staticmethod
    def _infill(x: torch.Tensor, slot: Slot) -> torch.Tensor:
        known_x = slot.value.get('value_0', None)

        if known_x is None:
            return x

        interp_w = slot.value.get('interp_w', None)
        if interp_w is not None:
            x = x + torch.matmul(interp_w, known_x - x)  # [B,T,T]x[B,T,D]

        known_w = slot.value.get('known_w', None)
        if known_w is not None:
            if len(known_w.shape) + 1 == len(x.shape):
                known_w = known_w.unsqueeze(-1)
            x = known_w * known_x + (1.0 - known_w) * x

        return x

    def build_clamp_fn(self, slot):
        def _clamp_fn(x):
            assert len(x.shape) == 3
            x = torch.cat([x[:, :, :6], rectify_rot6d(x[:, :, 6:])], dim=2)  # clamp
            x = self._infill(x, slot)
            return x

        return _clamp_fn

    def batch_decode(self, slot: Slot, outputs):
        value = torch.stack([o.feature for o in outputs], dim=0)
        bvh_motions = self._batch_convert_rot6d_to_bvh(
            value.detach().cpu().numpy(),
            orient_rotate=slot.value.get('rotate_y', None),
            trans_offset=slot.value.get('offset_xz', None),
        )
        for i, o in enumerate(outputs):
            assert (o.bvh_header is None) and (o.bvh_motion is None), "Sample already decoded."
            o.bvh_header = self.bvh_header
            o.bvh_motion = bvh_motions[i]
        return outputs

    def postprocess(self, outputs, **sample):
        target_slot = Slot.get_target_slot_from_sample(sample)
        return self.batch_decode(target_slot, outputs)

    def custom_reg_loss(self, slot: Slot, prediction, target, sample_weights):
        assert self.inbetween_args is not None, "The current custom regularization loss is for in-betweening only."
        return self._inbetween_loss(slot, prediction, target, sample_weights)

    def _inbetween_loss(self, slot: Slot, prediction, target, sample_weights):
        prediction = self._infill(prediction, slot)
        target = self._infill(target, slot)

        batch_size, seq_len, num_joints = prediction.shape
        num_joints = (num_joints - 6) // 6
        assert sample_weights.shape == (batch_size,)

        # local rotations
        pred_rot = rot6d_to_rotmat(prediction[:, :, 6:].view(batch_size, seq_len, num_joints, 6))
        true_rot = rot6d_to_rotmat(target[:, :, 6:].view(batch_size, seq_len, num_joints, 6))

        pred_trans = prediction[:, :, :3]
        true_trans = target[:, :, :3]

        # global rotations and global positions
        pred_rot, pred_pos = self.bvh_header.forward_kinematics(non_leaf_rotations=pred_rot, root_offsets=pred_trans)
        true_rot, true_pos = self.bvh_header.forward_kinematics(non_leaf_rotations=true_rot, root_offsets=true_trans)

        loss = F.l1_loss(pred_pos, true_pos, reduction='none').mean(dim=[2, 3])  # [B,T,J,3]->[B,T]
        loss += F.l1_loss(pred_rot, true_rot, reduction='none').mean(dim=[2, 3, 4])  # [B,T,J,3,3]->[B,T]

        weights = torch.logical_not(slot.value["masks"]).type_as(loss)
        weights *= 1.0 - slot.value['known_w']  # optimize only the unknown frames
        assert loss.shape == weights.shape
        loss = torch.sum(weights * loss, dim=-1) / torch.sum(weights, dim=-1)  # [B,T]->[B]
        loss = torch.mean(sample_weights * loss, dim=-1)
        return loss
