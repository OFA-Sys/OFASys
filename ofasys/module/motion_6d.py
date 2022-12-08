# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
import io
import math
import re
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import (
    Rotation as R,  # faster than transforms3d if (and only if) vectorized
)
from transforms3d.euler import euler2mat

from ofasys.utils.oss import oss_get


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rectify_rot6d(x):
    """
    Input:
        (B,6) Batch of *approximate* 6-D rotation representations
    Output:
        (B,6) Batch of *valid* 6-D rotation representations
    """
    a = x.reshape(-1, 3, 2)
    a1 = a[:, :, 0]
    a2 = a[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b = torch.stack((b1, b2), dim=-1)
    y = b.reshape(*x.shape)
    return y


def convert_bvh_to_rot6d(
    frames: np.ndarray, rotate_order='XYZ', scale_trans=1.0, scale_rot6d=1.0, aug_rot_mat=None
) -> np.ndarray:
    num_frames, num_joints = frames.shape
    num_joints = (num_joints - 3) // 3
    assert num_frames >= 2
    num_frames -= 1

    rot_mats = R.from_euler(rotate_order, frames[1:, 3:].reshape((num_frames * num_joints, 3)), degrees=True)
    rot_mats = rot_mats.as_matrix().reshape((num_frames, num_joints, 3, 3))

    velocity = (frames[1:, :3] - frames[:-1, :3]) * scale_trans
    velocity = np.matmul(rot_mats[:, 0].transpose(0, 2, 1), velocity[:, :, None]).squeeze(-1)

    rot_mats[:, 0] = np.matmul(aug_rot_mat[None, :, :], rot_mats[:, 0])

    pose_seq = np.concatenate(
        [velocity, scale_rot6d * rot_mats[:, :, :, :2].reshape((num_frames, num_joints * 6))], axis=-1
    )
    assert pose_seq.shape == (num_frames, 3 + 6 * num_joints)
    return pose_seq


def batch_convert_rot6d_to_bvh(
    trans_and_poses: torch.FloatTensor, rotate_order='XYZ', scale_trans=1.0, scale_rot6d=1.0
) -> np.ndarray:
    batch_size, seq_len, embed_dim = trans_and_poses.shape
    num_joints = (embed_dim - 3) // 6

    rot_mats = rot6d_to_rotmat(trans_and_poses[:, :, 3:].reshape(batch_size * seq_len * num_joints, 6) / scale_rot6d)
    rot_mats = rot_mats.view(batch_size, seq_len, num_joints, 3, 3).detach().cpu().numpy()

    velocity = trans_and_poses[:, :, :3].detach().cpu().numpy()
    velocity = np.matmul(rot_mats[:, :, 0], velocity[:, :, :, None]).squeeze(-1)

    eulers = R.from_matrix(rot_mats.reshape(batch_size * seq_len * num_joints, 3, 3))
    eulers = eulers.as_euler(rotate_order, degrees=True).reshape((batch_size, seq_len, num_joints * 3))

    bvh_frames = np.concatenate([np.cumsum(velocity, axis=1) / scale_trans, eulers], axis=-1)
    assert bvh_frames.shape == (batch_size, seq_len, 3 + 3 * num_joints)
    return bvh_frames


def discrete_fourier_transform(x_in_time, norm='ortho'):
    batch_size, seq_len, data_dim = x_in_time.shape
    dtype, device = x_in_time.dtype, x_in_time.device

    arange = torch.arange(seq_len, device=device)
    freqs = arange[: 1 + (seq_len // 2)]

    bases = (2 * math.pi / seq_len) * (arange.unsqueeze(0) * freqs.unsqueeze(-1)).to(dtype=dtype)  # [F,T]
    cos_freqs = freqs
    cos_bases = bases
    if seq_len % 2 == 0:
        sin_freqs = freqs[1:-1]
        sin_bases = bases[1:-1]
    else:
        sin_freqs = freqs[1:]
        sin_bases = bases[1:]

    x_in_freq = torch.cat(
        [
            torch.matmul(torch.cos(cos_bases).unsqueeze(0), x_in_time),
            torch.matmul(-torch.sin(sin_bases).unsqueeze(0), x_in_time),
        ],
        dim=1,
    )
    if norm == 'forward':
        x_in_freq = x_in_freq / seq_len
    elif norm == 'ortho':
        x_in_freq = x_in_freq / (seq_len**0.5)
    else:
        assert norm == 'backward'
    assert x_in_freq.shape == x_in_time.shape
    return x_in_freq, cos_freqs, sin_freqs


def inverse_discrete_fourier_transform(x_in_freq, norm='ortho'):
    batch_size, seq_len, data_dim = x_in_freq.shape
    dtype, device = x_in_freq.dtype, x_in_freq.device

    arange = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = (2 * math.pi / seq_len) * arange[: 1 + (seq_len // 2)]

    bases = freqs.unsqueeze(0) * arange.unsqueeze(-1)  # [T,F]
    cos_bases = torch.cos(bases)
    if seq_len % 2 == 0:
        cos_bases[:, 1:-1] *= 2
        sin_bases = -torch.sin(bases[:, 1:-1]) * 2
    else:
        cos_bases[:, 1:] *= 2
        sin_bases = -torch.sin(bases[:, 1:]) * 2
    bases = torch.cat([cos_bases, sin_bases], dim=-1)

    x_in_time = torch.matmul(bases.unsqueeze(0), x_in_freq)
    if norm == 'backward':
        x_in_time = x_in_time / seq_len
    elif norm == 'ortho':
        x_in_time = x_in_time / (seq_len**0.5)
    else:
        assert norm == 'forward'
    assert x_in_time.shape == x_in_freq.shape
    return x_in_time


# DFT_v2 and iDFT_v2 support masking, but are much slower and memory-hungrier than v1 due to batch matmul.
# We use norm='ortho' so that the output values' scales are roughly the same even when the lengths vary.
def discrete_fourier_transform_v2(x_in_time, padding_masks, norm='ortho'):
    batch_size, max_len, data_dim = x_in_time.shape
    dtype, device = x_in_time.dtype, x_in_time.device
    assert padding_masks.shape == (batch_size, max_len)

    w = torch.logical_not(padding_masks).to(dtype=dtype)  # [B,T/F]
    seq_lens = torch.sum(w, dim=-1).view(batch_size, 1, 1)  # [B,1,1]

    arange = torch.arange(max_len, dtype=dtype, device=device).view(1, 1, max_len)  # [1,1,T]
    freqs = (2 * math.pi) * arange.view(1, max_len, 1) / seq_lens  # [B,F,1]
    bases = arange * freqs  # [B,F,T]

    x_in_freq = torch.cat(
        [
            torch.bmm(torch.cos(bases) * w.unsqueeze(1) * w.unsqueeze(-1), x_in_time),  # [B,F,T]x[B,T,D]->[B,F,D]
            torch.bmm(-torch.sin(bases) * w.unsqueeze(1) * w.unsqueeze(-1), x_in_time),  # [B,F,T]x[B,T,D]->[B,F,D]
        ],
        dim=-1,
    )  # [B,F,2*D]

    if norm == 'forward':
        x_in_freq = x_in_freq / seq_lens
    elif norm == 'ortho':
        x_in_freq = x_in_freq / (seq_lens**0.5)
    else:
        assert norm == 'backward'

    assert x_in_freq.shape == (batch_size, max_len, data_dim * 2)
    return x_in_freq, freqs.reshape(batch_size, max_len)


# It only returns the real part and ignores the imaginary part.
def inverse_discrete_fourier_transform_v2(x_in_freq, padding_masks, norm='ortho'):
    batch_size, max_len, data_dim = x_in_freq.shape
    assert data_dim % 2 == 0
    data_dim //= 2
    dtype, device = x_in_freq.dtype, x_in_freq.device

    w = torch.logical_not(padding_masks).to(dtype=dtype)  # [B,T/F]
    seq_lens = torch.sum(w, dim=-1).view(batch_size, 1, 1)  # [B,1,1]

    arange = torch.arange(max_len, dtype=dtype, device=device).view(1, max_len, 1)  # [1,T,1]
    freqs = (2 * math.pi) * arange.view(1, 1, max_len) / seq_lens  # [1,1,F]*[B,1,1]->[B,1,F]
    bases = arange * freqs  # [B,T,F]

    x_cos, x_sin = x_in_freq[:, :, :data_dim], x_in_freq[:, :, data_dim:]  # [B,F,D]
    x_in_time = torch.bmm(torch.cos(bases) * w.unsqueeze(1) * w.unsqueeze(-1), x_cos) + torch.bmm(
        -torch.sin(bases) * w.unsqueeze(1) * w.unsqueeze(-1), x_sin
    )  # [B,T,D]

    if norm == 'backward':
        x_in_time = x_in_time / seq_lens
    elif norm == 'ortho':
        x_in_time = x_in_time / (seq_lens**0.5)
    else:
        assert norm == 'forward'

    assert x_in_time.shape == (batch_size, max_len, data_dim)
    return x_in_time


# Caution: torch.sqrt(x) and torch.atan2(_, x) can cause NaN/Inf gradients when x = 0.
def discrete_fourier_transform_v3(x_in_time, get_unnorm_sqr_amp=False):
    batch_size, seq_len, data_dim = x_in_time.shape
    dtype, device = x_in_time.dtype, x_in_time.device

    num_amplitudes = 1 + (seq_len // 2)
    num_phases = (seq_len - 1) // 2
    assert num_amplitudes + num_phases == seq_len

    arange = torch.arange(seq_len, device=device)
    freqs = arange[:num_amplitudes]
    bases = (2 * math.pi / seq_len) * (arange.unsqueeze(0) * freqs.unsqueeze(-1)).to(dtype=dtype)  # [F,T]

    x_cos = torch.matmul(torch.cos(bases).unsqueeze(0), x_in_time)  # [B,F,D]
    x_sin = torch.matmul(-torch.sin(bases).unsqueeze(0), x_in_time)  # [B,F,D]

    if get_unnorm_sqr_amp:
        return x_cos.square() + x_sin.square()

    amplitudes = torch.sqrt(x_cos.square() + x_sin.square()) / seq_len
    phases = torch.atan2(x_sin[:, 1 : (1 + num_phases)], x_cos[:, 1 : (1 + num_phases)])
    return amplitudes, phases


def geodesic_loss(input, target, reduction='none', eps=1e-7):
    # input.shape == [..., 3, 3]
    # target.shape == [..., 3, 3]
    out_shape = input.shape[:-2]
    input = input.reshape(-1, 3, 3)
    target = target.reshape(-1, 3, 3)
    r_diffs = input @ target.permute(0, 2, 1)
    traces = r_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
    dists = dists.reshape(*out_shape)
    if reduction == 'sum':
        return dists.sum()
    elif reduction == 'mean':
        return dists.mean()
    assert reduction == 'none'
    return dists


if __name__ == '__main__':

    def _test_dft_idft():
        norm_mode = 'ortho'
        a = torch.randn(np.random.randint(1, 128), np.random.randint(1, 128), np.random.randint(1, 128))
        x, _, _ = discrete_fourier_transform(a, norm_mode)
        y = torch.view_as_real(torch.fft.rfft(a, dim=1, norm=norm_mode))
        if a.shape[1] % 2 == 0:
            y = torch.cat([y[:, :, :, 0], y[:, 1:-1, :, 1]], dim=1)
        else:
            y = torch.cat([y[:, :, :, 0], y[:, 1:, :, 1]], dim=1)
        b = inverse_discrete_fourier_transform(x, norm_mode)
        print(a.shape, x.shape, y.shape, b.shape)
        print((x - y).abs().max())
        print((a - b).abs().max())

    def _test_dft_linear():
        norm_mode = 'forward'
        a = torch.randn(np.random.randint(1, 128), np.random.randint(1, 128), np.random.randint(1, 128))
        w = torch.nn.Linear(a.shape[-1], 10, bias=False)
        x, _, _ = discrete_fourier_transform(w(a), norm_mode)
        xx, _, _ = discrete_fourier_transform(a, norm_mode)
        xx = w(xx)
        print((x - xx).abs().max())

    def _test_dft_add():
        norm_mode = 'ortho'
        a = torch.randn(np.random.randint(1, 128), np.random.randint(1, 128), np.random.randint(1, 128))
        b = torch.randn(*a.shape)
        x, _, _ = discrete_fourier_transform(a, norm_mode)
        y, _, _ = discrete_fourier_transform(b, norm_mode)
        xy = inverse_discrete_fourier_transform(x + y, norm_mode)
        ab = a + b
        print((xy - ab).abs().max())

    def _test_dft_mask():
        norm_mode = 'ortho'
        a = torch.zeros(np.random.randint(1, 10), np.random.randint(1, 10), np.random.randint(1, 10))
        m = torch.ones(*a.shape[:2], dtype=torch.bool)
        for i in range(a.shape[0]):
            ai = torch.randn(np.random.randint(1, a.shape[1] + 1), a.shape[2])
            print(i, torch.fft.fft(ai, dim=0, norm=norm_mode))
            a[i, : ai.shape[0]] = ai
            m[i, : ai.shape[0]] = False
        x, _ = discrete_fourier_transform_v2(a, m, norm_mode)
        y = inverse_discrete_fourier_transform_v2(x, m, norm_mode)
        print(x)
        assert y.shape == a.shape
        print((a - y).abs().max())

    def _test_dft_in_mag_phase():
        n = np.random.randint(1, 100)
        a = torch.randn(n)
        f = torch.view_as_real(torch.fft.fft(a))
        m = ((f[:, 0] ** 2 + f[:, 1] ** 2) ** 0.5) / n
        p = torch.atan2(f[:, 1], f[:, 0])
        print('n', n)
        print('amplitude:', m)
        print('phase:', p)
        print('input:', a)
        print(discrete_fourier_transform_v3(a.reshape(1, n, 1)))
        xs = []
        for t in range(n):
            x = 0.0
            for k in range(n):
                x += m[k] * torch.cos(2 * math.pi * t / n * k + p[k])
            xs.append(x)
        xs = torch.stack(xs)
        print('reconstruct:', xs)
        print('error:', (a - xs).abs().max())

    _test_dft_in_mag_phase()


def convert_to_bvh(frames, fps=30):
    if isinstance(frames, np.ndarray):
        frames = frames.tolist()

    fin = oss_get('oss://ofasys/data/motion_diffusion_data/smplh_bvh_header.bvh')
    with BytesIO(fin.read()) as bio:
        bvh = io.TextIOWrapper(bio, encoding='utf-8').read()
    del fin  # close
    bvh += '\n' + 'MOTION'
    bvh += '\n' + 'Frames: ' + str(len(frames))
    bvh += '\n' + 'Frame Time: %f' % (1.0 / fps)
    for f in frames:
        bvh += '\n' + ' '.join(str(x) for x in f)
    return BvhObject(text=bvh)


class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []
        self.index = None


class BvhObject:
    def __init__(self, path=None, text=None, joints=None):
        self.root = None
        self.joints = []
        self.frames = []
        self.fps = 30

        if joints is not None:
            assert path is None
            assert text is None
            assert joints[0].parent is None
            for j in joints[1:]:
                assert j.parent is not None
            self.root = joints[0]
            self.joints = joints
            return

        if path is not None:
            assert text is None
            with open(path, 'r') as f:
                text = f.read()
        assert text is not None
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self._parse_motion(motion)
        for i, joint in enumerate(self.joints):
            joint.index = i

    def _parse_hierarchy(self, text):
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

    def _parse_motion(self, text):
        num_frames = 0
        self.frames = []
        for line in re.split('\\s*\\n+\\s*', text):
            if line != '':
                words = re.split('\\s+', line)
                if line.startswith("Frame Time:"):
                    self.fps = round(1 / float(words[2]))
                elif line.startswith("Frames:"):
                    num_frames = int(words[1])
                else:
                    self.frames.append([float(w) for w in words])
        assert num_frames > 0
        assert len(self.frames) == num_frames
        for i in range(1, len(self.frames)):
            assert len(self.frames[i - 1]) == len(self.frames[i])

    def save_as_bvh(self, path):
        with open(path, 'w') as fout:
            print('HIERARCHY', file=fout)

            def _write_header(joint, num_tab=0):
                if joint.parent is None:
                    assert num_tab == 0
                    print('\t' * num_tab + 'ROOT ' + joint.name, file=fout)
                elif len(joint.children) > 0:
                    print('\t' * num_tab + 'JOINT ' + joint.name, file=fout)
                else:
                    print('\t' * num_tab + 'End Site', file=fout)
                print('\t' * num_tab + '{', file=fout)
                print('\t' * (num_tab + 1) + 'OFFSET ' + ' '.join([str(x) for x in joint.offset]), file=fout)
                if len(joint.children) > 0:
                    print(
                        '\t' * (num_tab + 1) + 'CHANNELS %d ' % (len(joint.channels)) + ' '.join(joint.channels),
                        file=fout,
                    )
                    for child in joint.children:
                        _write_header(child, num_tab + 1)
                print('\t' * num_tab + '}', file=fout)

            _write_header(self.root)

            print('MOTION', file=fout)
            print('Frames: %d' % len(self.frames), file=fout)
            print('Frame Time: %.6f' % (1.0 / self.fps), file=fout)
            for i, f in enumerate(self.frames):
                assert (i == 0) or (len(self.frames[i - 1]) == len(self.frames[i]))
                print(' '.join([str(v) for v in f]), file=fout)

    def normalize_scale(self, divide_by=None):
        if divide_by is None:
            divide_by = max([(b.offset**2).sum() ** 0.5 for b in self.joints]) * 2.0
        for u in self.joints:
            u.offset /= divide_by
        for f in self.frames:
            f[0] = f[0] / divide_by
            f[1] = f[1] / divide_by
            f[2] = f[2] / divide_by
        return self

    def center_around_origin(self):
        mean_tx = sum(f[0] for f in self.frames) / len(self.frames)
        mean_ty = sum(f[1] for f in self.frames) / len(self.frames)
        mean_tz = sum(f[2] for f in self.frames) / len(self.frames)
        for f in self.frames:
            f[0] = f[0] - mean_tx
            f[1] = f[1] - mean_ty
            f[2] = f[2] - mean_tz
        return self

    def save_as_gif(self, path):
        bvh = copy.deepcopy(self).center_around_origin().normalize_scale()
        plot_all_frames(bvh_object=bvh, gif_path=path)


def pose_fk(bvh_joints, bvh_frame):
    floats = copy.deepcopy(bvh_frame)[::-1]
    positions = np.zeros((len(bvh_joints), 3))
    rotations = np.zeros((len(bvh_joints), 3, 3))

    for idx, joint in enumerate(bvh_joints):
        assert joint.index == idx
        if joint.parent is None:
            parent_position = np.zeros((3,))
            parent_rotation = np.eye(3)
        else:
            parent_position = positions[joint.parent.index]
            parent_rotation = rotations[joint.parent.index]
        positions[joint.index] = parent_position + parent_rotation.dot(joint.offset)

        rel_position = np.zeros(3)
        for channel in joint.channels:
            if channel == "Xposition":
                assert joint.parent is None
                rel_position[0] += floats.pop()
            elif channel == "Yposition":
                assert joint.parent is None
                rel_position[1] += floats.pop()
            elif channel == "Zposition":
                assert joint.parent is None
                rel_position[2] += floats.pop()
        positions[joint.index] += rel_position

        rel_rotation = np.eye(3)
        for channel in joint.channels:
            if channel == "Xrotation":
                rel_rotation = rel_rotation.dot(euler2mat(np.deg2rad(floats.pop()), 0.0, 0.0))
            elif channel == "Yrotation":
                rel_rotation = rel_rotation.dot(euler2mat(0.0, np.deg2rad(floats.pop()), 0.0))
            elif channel == "Zrotation":
                rel_rotation = rel_rotation.dot(euler2mat(0.0, 0.0, np.deg2rad(floats.pop())))
        rotations[joint.index] = parent_rotation.dot(rel_rotation)

    assert len(floats) == 0
    return positions


def plot_frame(bvh_joints, bvh_frame, ax):
    p = pose_fk(bvh_joints, bvh_frame)
    xs, ys, zs = p[:, 0], -p[:, 2], p[:, 1]
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-2, 2)
    ret = [ax.scatter(xs=xs, ys=ys, zs=zs)]
    for u in bvh_joints:
        for v in u.children:
            i = u.index
            j = v.index
            ret.append(ax.plot(xs=[xs[i], xs[j]], ys=[ys[i], ys[j]], zs=[zs[i], zs[j]]))
    return tuple(ret)


def plot_all_frames(bvh_object, gif_path):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def _animate(i):
        return plot_frame(bvh_object.joints, bvh_object.frames[i], ax)

    fps = 30
    ani = FuncAnimation(fig, _animate, interval=1000 / fps, blit=False, repeat=True, frames=len(bvh_object.frames))
    ani.save(gif_path, dpi=300, writer=PillowWriter(fps=fps))
