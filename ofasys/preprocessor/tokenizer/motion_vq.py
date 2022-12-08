# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation, Slerp

from ofasys.module.taming.models.vqgan import VQModel
from ofasys.module.taming.modules.diffusionmodules.model import (
    AttnBlock,
    Downsample,
    Normalize,
    ResnetBlock,
    Upsample,
    nonlinearity,
)
from ofasys.utils.file_utils import cached_path


class MotionEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=32,
        z_channels=256,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dilation=2, dropout=dropout
        )
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class MotionDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=32,
        z_channels=256,
        give_pre_end=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dilation=2, dropout=dropout
        )
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)

        self.conv_vec1 = torch.nn.Conv2d(block_in, 64, kernel_size=3, stride=1, padding=1)
        self.conv_vec2 = torch.nn.Conv2d(64, 3, kernel_size=(1, 24), stride=1, padding=0)

        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h_pose = self.conv_out(h)
        h_vec = self.conv_vec2(nonlinearity(self.conv_vec1(h))) / 10

        return h_pose, h_vec


class MotionVQModel(VQModel):
    def __init__(
        self,
        n_embed,
        embed_dim,
        image_key="image",
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__(None, None, n_embed, embed_dim, remap=remap, sane_index_shape=sane_index_shape)
        self.image_key = image_key  ## B C T N
        self.encoder = MotionEncoder(attn_resolutions=[1000], ch_mult=(1, 2, 4))
        self.decoder = MotionDecoder(attn_resolutions=[1000], ch_mult=(1, 2, 4))
        self.quant_conv = torch.nn.Conv2d(512, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 256, 1)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec, trans = self.decoder(quant)
        return dec, trans

    def decoder_random(self, z):
        z_q = self.quantize.embedding(z)
        batch = z_q.shape[0]
        z_q = z_q.view(batch, -1, 6, 512)
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        dec, trans = self.decode(z_q)
        return dec, trans

    def decode_code(self, code_b):
        quant_b = self.quantize.embedding(code_b).view(code_b.shape[0], 4, 6, -1)
        quant_b = rearrange(quant_b, 'b h w c -> b c h w').contiguous()
        dec, trans = self.decode(quant_b)
        return dec, trans

    def forward(self, input):
        quant, diff, info = self.encode(input)
        dec, trans = self.decode(quant)
        return dec, trans, diff, info[2].view(dec.shape[0], -1)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.

    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(  # pyre-ignore[16]
        batch_dim + (4,)
    )


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


class MotionVQGANTokenizer(object):
    def __init__(self, vqgan_model_path, n_embed=1024, embed_dim=512):
        vqgan = MotionVQModel(n_embed, embed_dim)

        local_model_path = cached_path(vqgan_model_path)
        sd = torch.load(local_model_path, map_location='cpu')["vqgan"]
        ori_keys = list(sd.keys())
        for k in ori_keys:
            if k.startswith('module.'):
                sd[k[7:]] = sd[k]
                del sd[k]
        vqgan.load_state_dict(sd)

        for k, v in vqgan.named_parameters():
            v.requires_grad = False
        self.vqgan = vqgan
        # use cpu for saving gpu memory
        self.device = 'cpu'
        # self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.vqgan.to(self.device)
        self.vqgan.eval()

    def decode(self, tokens, **kwargs):
        batch_size = tokens.size()[0]
        uniq_ids = kwargs.get('inputs', None)
        ret = []
        for i in range(batch_size):
            name = uniq_ids[i] if uniq_ids else i
            inputs_ = tokens[i].tolist()
            lens = (len(inputs_) // 24) * 24
            inputs = torch.Tensor(inputs_[:lens]).long().view(1, 1, -1, 6)
            xrec, trans = self.vqgan.decoder_random(inputs)
            trans = trans.permute(0, 2, 3, 1).reshape(-1, 1, 3)
            for step_ in range(1, trans.shape[0]):
                trans[step_, :, :] = trans[step_ - 1, :, :] + trans[step_, :, :] - trans[step_ - 1, :, :] / 10
            xrec = xrec.permute(0, 2, 3, 1).reshape(-1, 24, 3)
            poses = rodrigues(xrec[::, :, :].reshape(-1, 3)).reshape(-1, 24, 3, 3).data.cpu().numpy()
            seq_len = poses.shape[0]
            interp_seq_len = seq_len * 4
            times = np.arange(seq_len, dtype=poses.dtype)
            interp_times = np.arange(interp_seq_len, dtype=poses.dtype)
            interp_times *= (seq_len - 1.0) / (interp_seq_len - 1.0)
            trans = trans.data.cpu().numpy()
            interp_trans = []
            for j in range(3):
                interp_trans.append(np.interp(x=interp_times, xp=times, fp=trans[:, 0, j]))
            interp_trans = np.stack(interp_trans, axis=1)
            interp_poses = []
            for j in range(poses.shape[1]):
                key_rots = Rotation.from_matrix(poses[:, j])
                slerp = Slerp(times, key_rots)
                interp_poses.append(slerp(interp_times).as_matrix())
            interp_poses = torch.from_numpy(np.stack(interp_poses, axis=1)).float()
            xrec = matrix_to_axis_angle(interp_poses).reshape(-1, 72)
            poses = xrec.data.cpu().numpy()
            betas = torch.zeros(10).data.numpy()
            trans = interp_trans
            ret.append({'name': name, 'poses': poses, 'betas': betas, 'trans': trans})
        return ret
