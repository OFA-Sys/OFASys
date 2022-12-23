# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
import logging
import re
from io import BytesIO, TextIOWrapper
from typing import List, Optional

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from ofasys.utils.oss import oss_get

logger = logging.getLogger(__name__)


def rotmat_to_rot6d(x, contingent=True):
    x_shape = x.shape
    assert x_shape[-2:] == (3, 3)
    if contingent:
        if isinstance(x, np.ndarray):
            x = np.concatenate([x[..., 0], x[..., 1]], axis=-1)
        else:
            x = torch.cat([x[..., 0], x[..., 1]], dim=-1)
    else:
        x = x[..., :2].reshape(*x_shape[:-2], 6)
    assert x.shape == (*x_shape[:-2], 6)
    return x


def rot6d_to_rotmat(x, contingent=True):
    x_shape = x.shape
    assert x_shape[-1] == 6
    if contingent:
        x = x.reshape(*x_shape[:-1], 2, 3)
        a1 = x[..., 0, :]
        a2 = x[..., 1, :]
    else:
        x = x.reshape(*x_shape[:-1], 3, 2)
        a1 = x[..., 0]
        a2 = x[..., 1]
    if isinstance(x, np.ndarray):
        b1 = a1 / np.linalg.norm(a1, ord=2, axis=-1, keepdims=True)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, ord=2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)
        x = np.stack((b1, b2, b3), axis=-1)
    else:
        b1 = F.normalize(a1, p=2, dim=-1)
        b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, p=2, dim=-1)
        b3 = torch.cross(b1, b2)
        x = torch.stack((b1, b2, b3), dim=-1)
    assert x.shape == (*x_shape[:-1], 3, 3)
    return x


def rectify_rot6d(x, contingent=True):
    x_shape = x.shape
    x = x.reshape(-1, 6)
    x = rotmat_to_rot6d(rot6d_to_rotmat(x, contingent), contingent)
    x = x.reshape(*x_shape)
    return x


# Modified from https://github.com/boreshkinai/delta-interpolator.
# performs differentiable forward kinematics
# local_rotations: local rotation matrices of each joint in the batch (B, J, 3, 3)
# local_offset: local position offset of each joint in the batch (B, J, 3).
#               This corresponds to the offset with respect to the parent joint when no rotation is applied
# level_joints: a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
#               each element of the list is a list transform indexes
#               for instance level_joints[3] contains a list of indices of all the joints that are 3 levels
#               below the root joints the indices should match these of the provided local rotations
#               for instance if level_joints[4, 1] is equal to 7, it means that local_rotations[:, 7, :, :]
#               contains the local rotation matrix of a joint that is 4 levels deeper than the root joint
#               (ie there are 4 joints above it in the hierarchy)
# level_joint_parents: similar to level_transforms but contains the parent indices
#                      for instance if level_joint_parents[4, 1] is equal to 5, it means that
#                      local_rotations[:, 5, :, :] contains the local rotation matrix of the parent joint of
#                      the joint contained in level_joints[4, 1]
def forward_kinematics(
    local_rotations: torch.Tensor,
    local_offsets: torch.Tensor,
    level_joints: List[List[int]],
    level_joint_parents: List[List[int]],
):
    batch_size, num_joints, _, _ = local_rotations.shape
    assert local_rotations.shape == (batch_size, num_joints, 3, 3)
    assert local_offsets.shape == (batch_size, num_joints, 3)

    # Note: PyTorch's CopySlices is differentiable.

    local_transforms = torch.zeros(
        batch_size, num_joints, 4, 4, dtype=local_rotations.dtype, device=local_rotations.device
    )
    local_transforms[:, :, :3, :3] = local_rotations
    local_transforms[:, :, :3, 3] = local_offsets
    local_transforms[:, :, 3, 3] = 1.0

    world_transforms = torch.zeros_like(local_transforms)
    world_transforms[:, level_joints[0]] = local_transforms[:, level_joints[0]]
    for level in range(1, len(level_joints)):
        parent_bone_indices = level_joint_parents[level]
        local_bone_indices = level_joints[level]
        parent_level_transforms = world_transforms[:, parent_bone_indices]
        local_level_transforms = local_transforms[:, local_bone_indices]
        world_transforms[:, local_bone_indices] = torch.matmul(parent_level_transforms, local_level_transforms)

    world_rotations = world_transforms[..., 0:3, 0:3]  # [B,J,3,3]
    world_offsets = world_transforms[..., 0:3, 3]  # [B,J,3]
    return world_rotations, world_offsets


class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent: BvhJoint = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children: List[BvhJoint] = []
        self.index = None


class BvhHeader:
    def __init__(self, path=None, text=None):
        self.root: Optional[BvhJoint] = None
        self.joints: List[BvhJoint] = []

        if path is not None:
            assert text is None
            if path.startswith("oss://"):
                fin = oss_get(path)
                with TextIOWrapper(BytesIO(fin.read())) as bio:
                    text = bio.read()
                del fin  # close
            else:
                with open(path, 'r') as f:
                    text = f.read()
        self.header_text = text.split("MOTION")[0]
        self._parse_header(self.header_text)

        i = 0
        for j in self.joints:
            if len(j.children) > 0:
                j.index = i
                i += 1
        self.n_inner_joints = i
        for j in self.joints:
            if len(j.children) == 0:
                j.index = i
                i += 1
        self.n_joints = len(self.joints)
        assert self.root == self.joints[0]

        level_joints = []
        level_joint_parents = []

        def _make_levels(joint, level):
            if level >= len(level_joints):
                level_joints.append([])
            level_joints[level].append(joint.index)
            if level >= len(level_joint_parents):
                level_joint_parents.append([])
            if level > 0:
                level_joint_parents[level].append(joint.parent.index)
            for child in joint.children:
                _make_levels(child, level + 1)

        _make_levels(self.root, 0)

        joint_offsets = torch.zeros(len(self.joints), 3)
        for j in self.joints:
            joint_offsets[j.index] = torch.from_numpy(j.offset.astype(np.float32))

        self.level_joints = level_joints
        self.level_joint_parents = level_joint_parents
        self.joint_offsets = joint_offsets

    def _parse_header(self, text):
        joint_stack = []
        for line in re.split('\\s*\\n+\\s*', text):
            words = re.split('\\s+', line)
            instruction = words[0]
            if instruction in ("JOINT", "ROOT", "End"):
                if instruction == "End":
                    name = joint_stack[-1].name + "_end"
                else:
                    name = words[1]
                if instruction == "ROOT":
                    parent = None
                else:
                    parent = joint_stack[-1]
                joint = BvhJoint(name=name, parent=parent)
                if parent is not None:
                    parent.children.append(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
                self.joints.append(joint)
            elif instruction == "CHANNELS":
                assert int(words[1]) == len(words) - 2
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == '}':
                joint_stack.pop()

    def get_rotation_order(self):
        # https://math.stackexchange.com/questions/1137745/proof-of-the-extrinsic-to-intrinsic-rotation-transform
        order = ''.join(a[0].lower() for a in self.joints[0].channels[3:])
        return order.upper()

    def forward_kinematics(self, non_leaf_rotations: torch.Tensor, root_offsets: Optional[torch.Tensor] = None):
        batch_size, n_frames, n_inner_joints, _, _ = non_leaf_rotations.shape
        dtype, device = non_leaf_rotations.dtype, non_leaf_rotations.device
        assert n_inner_joints == self.n_inner_joints
        assert non_leaf_rotations.shape == (batch_size, n_frames, n_inner_joints, 3, 3)

        self.joint_offsets = self.joint_offsets.to(dtype=dtype, device=device)

        local_offsets = self.joint_offsets.unsqueeze(0).repeat(batch_size * n_frames, 1, 1)  # [B,J,3]
        if root_offsets is not None:
            assert root_offsets.shape == (batch_size, n_frames, 3)
            local_offsets[:, 0] = root_offsets.view(batch_size * n_frames, 3)

        local_rotations = (
            torch.eye(3, dtype=dtype, device=device)
            .view(1, 1, 3, 3)
            .repeat(batch_size * n_frames, self.n_joints, 1, 1)
        )
        local_rotations[:, :n_inner_joints] = non_leaf_rotations.view(batch_size * n_frames, n_inner_joints, 3, 3)

        global_rotations, global_positions = forward_kinematics(
            local_rotations, local_offsets, self.level_joints, self.level_joint_parents
        )
        global_rotations = global_rotations.view(batch_size, n_frames, self.n_joints, 3, 3)
        # global_rotations = global_rotations[:, :, :n_inner_joints]
        global_positions = global_positions.view(batch_size, n_frames, self.n_joints, 3)
        return global_rotations, global_positions

    def _get_joint_positions(self, bvh_motion: np.ndarray) -> np.ndarray:
        assert len(bvh_motion.shape) == 2, "Require [num_frames, data_dim], not [batch_size, num_frames, data_dim]."
        seq_len, num_joints = bvh_motion.shape
        num_joints = (num_joints - 3) // 3
        rot_order = self.get_rotation_order()

        rotations = R.from_euler(rot_order, bvh_motion[:, 3:].reshape((seq_len * num_joints, 3)), degrees=True)
        rotations = rotations.as_matrix().reshape((seq_len, num_joints, 3, 3))
        _, joint_positions = self.forward_kinematics(
            non_leaf_rotations=torch.from_numpy(rotations).view(1, seq_len, num_joints, 3, 3),
            root_offsets=torch.from_numpy(bvh_motion[:, :3]).view(1, seq_len, 3),
        )
        joint_positions = joint_positions.squeeze(0).detach().cpu().numpy()
        return joint_positions

    def save_as_gif(self, bvh_motion: np.ndarray, path: str, current_fps: float = 30, gauss_filter_sigma: float = 1.0):
        joint_positions = self._get_joint_positions(bvh_motion)
        seq_len, _ = bvh_motion.shape

        if gauss_filter_sigma > 0.0:
            joint_positions = gaussian_filter1d(
                joint_positions, sigma=gauss_filter_sigma, order=0, axis=0, mode='nearest'
            )

        from matplotlib import pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def _animate(i):
            p = joint_positions[i]
            xs, ys, zs = p[:, 0], -p[:, 2], p[:, 1]
            ax.clear()
            ax.set_xlim(-3, 3)
            ax.set_ylim(-4, 4)
            ax.set_zlim(-2, 2)
            ret = [ax.scatter(xs=xs, ys=ys, zs=zs)]
            for u in self.joints:
                for v in u.children:
                    i = u.index
                    j = v.index
                    ret.append(ax.plot(xs=[xs[i], xs[j]], ys=[ys[i], ys[j]], zs=[zs[i], zs[j]]))
            return tuple(ret)

        ani = FuncAnimation(fig, _animate, interval=1000 / current_fps, blit=False, repeat=True, frames=seq_len)
        # noinspection PyTypeChecker
        ani.save(path, dpi=100, writer=PillowWriter(fps=current_fps))

    def save_as_bvh(
        self,
        bvh_motion: np.ndarray,
        path: str,
        current_fps: float = 30,
        target_fps: float = 30,
        gauss_filter_sigma: float = 1.0,
    ):
        assert len(bvh_motion.shape) == 2, "Require [num_frames, data_dim], not [batch_size, num_frames, data_dim]."
        bvh_motion = self._interp_and_filter(bvh_motion, current_fps, target_fps, gauss_filter_sigma)
        with open(path, 'w') as fout:
            print(self.header_text, file=fout)
            print('MOTION', file=fout)
            print('Frames: %d' % bvh_motion.shape[0], file=fout)
            print('Frame Time: %.6f' % (1.0 / target_fps), file=fout)
            for i in range(bvh_motion.shape[0]):
                print(' '.join([str(v) for v in bvh_motion[i]]), file=fout)

    def _interp_and_filter(
        self, bvh_motion: np.ndarray, current_fps: float, target_fps: float, gauss_filter_sigma: float
    ) -> np.ndarray:
        if bvh_motion.shape[0] < 2:
            logger.warning('Interpolation and filtering require at least two frames.')
            return bvh_motion

        seq_len, num_joints = bvh_motion.shape
        num_joints = (num_joints - 3) // 3
        rot_order = self.get_rotation_order()

        trans = bvh_motion[:, :3]
        poses = R.from_euler(rot_order, bvh_motion[:, 3:].reshape((seq_len * num_joints, 3)), degrees=True)
        poses = poses.as_matrix().reshape((seq_len, num_joints, 3, 3))

        times = np.arange(seq_len, dtype=poses.dtype)
        interp_seq_len = round(seq_len * target_fps / current_fps)
        assert interp_seq_len // seq_len * seq_len == interp_seq_len
        interp_times = np.arange(interp_seq_len, dtype=poses.dtype)
        interp_times *= (seq_len - 1.0) / (interp_seq_len - 1.0)

        # To prevent "ValueError: Interpolation times must be within the range..." caused by floats.
        interp_times[0] = max(interp_times[0], times[0])
        interp_times[-1] = min(interp_times[-1], times[-1])

        interp_trans = []
        for j in range(3):
            interp_trans.append(np.interp(x=interp_times, xp=times, fp=trans[:, j]))
        interp_trans = np.stack(interp_trans, axis=1)
        assert interp_trans.shape == (interp_seq_len, 3)

        interp_poses = []
        for j in range(num_joints):
            key_rots = R.from_matrix(poses[:, j])
            interp_poses.append(Slerp(times, key_rots)(interp_times).as_matrix())
        interp_poses = np.stack(interp_poses, axis=1)
        assert interp_poses.shape == (interp_seq_len, num_joints, 3, 3)

        if gauss_filter_sigma > 0.0:

            def _filter(x):
                return gaussian_filter1d(x, sigma=gauss_filter_sigma, order=0, axis=0, mode='nearest')

            interp_trans = _filter(interp_trans)
            interp_poses = _filter(interp_poses)

        interp_poses = R.from_matrix(interp_poses.reshape((interp_seq_len * num_joints, 3, 3)))
        interp_poses = interp_poses.as_euler(rot_order, degrees=True).reshape((interp_seq_len, num_joints * 3))

        new_motion = np.concatenate([interp_trans, interp_poses], axis=-1)
        assert new_motion.shape == (interp_seq_len, 3 + num_joints * 3)
        return new_motion
