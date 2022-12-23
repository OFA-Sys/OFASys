# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ofasys import ModalityType
from ofasys.configure import ConfigStore, register_config
from ofasys.module import resnet50_backbone  # noqa
from ofasys.module import resnet101_backbone  # noqa
from ofasys.module import resnet152_backbone  # noqa
from ofasys.module import Embedding
from ofasys.preprocessor import Dictionary

from .base import AdaptorOutput, BaseAdaptor, BaseAdaptorConfig, Slot
from .image_resnet import ImageResnetAdaptor


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


@dataclass
class VideoImageSequenceAdaptorConfig(BaseAdaptorConfig):
    token_bucket_size: int = field(
        default=256,
        metadata={"help": "token bucket size"},
    )


def make_video_bucket_position(bucket_size, max_position=8192):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))
    log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid
    log_pos = log_pos.int()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos * sign).long()
    return bucket_pos + bucket_size - 1


@register_config("ofasys.adaptor", "video_image_sequence", VideoImageSequenceAdaptorConfig)
class VideoImageSequenceAdaptor(BaseAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: VideoImageSequenceAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)
        self.embed_frame_positions = Embedding(1024 + 1, cfg.embed_dim, zero_init=True)
        video_num_rel_dis = 2 * cfg.token_bucket_size - 1
        video_rp_bucket = make_video_bucket_position(cfg.token_bucket_size, 1024)

        num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
        self.video_rel_pos_table_list = nn.ModuleList(
            [Embedding(video_num_rel_dis, cfg.num_attention_heads, zero_init=True) for _ in range(num_rel_pos_tables)]
        )
        self.register_buffer("video_rp_bucket", video_rp_bucket)

        if 'image_resnet' not in self.general_adaptor.name2adaptor and self.is_src:
            self.general_adaptor.name2adaptor['image_resnet'] = (
                ConfigStore()
                .get('ofasys.adaptor', 'image_resnet')
                .target(
                    self.embed_tokens,
                    self.dictionary,
                    self.is_src,
                    self.general_adaptor,
                    getattr(self.general_adaptor.cfg.adaptor, 'image_resnet'),
                )
            )
            setattr(self.general_adaptor, 'image_resnet', self.general_adaptor.name2adaptor['image_resnet'])

    def train(self, mode=True):
        super().train(mode)

    def get_image_resnet_adaptor(self) -> ImageResnetAdaptor:
        image_resnet_adaptor: ImageResnetAdaptor = self.general_adaptor.name2adaptor['image_resnet']
        assert image_resnet_adaptor is not None
        return image_resnet_adaptor

    def get_rel_pos_bias(self, batch_size, seq_length, idx, **kwargs):
        rp_bucket = self.video_rp_bucket[:seq_length, :seq_length]
        values = F.embedding(rp_bucket, self.video_rel_pos_table_list[idx].weight)
        return values

    def get_clip_videos_info(self, clip_videos: torch.Tensor):
        image_resnet_adaptor = self.get_image_resnet_adaptor()
        device = clip_videos.device
        with torch.no_grad():
            clip_videos = clip_videos.transpose(1, 2)
        batch_size, frames_per_video = clip_videos.size(0), clip_videos.size(1)
        image_embed_full_resolution: torch.Tensor = image_resnet_adaptor.embed_images(
            clip_videos.reshape(-1, clip_videos.size(2), clip_videos.size(3), clip_videos.size(4))
        )

        h, w = image_embed_full_resolution.shape[-2:]
        image_embed_full_resolution = image_embed_full_resolution.view(
            image_embed_full_resolution.size(0), image_embed_full_resolution.size(1), -1
        ).transpose(1, 2)
        image_embed = image_embed_full_resolution

        image_num_patches: int = h * w
        video_num_patches: int = image_num_patches * frames_per_video

        # This is somewhat ugly and may lead to bug if we really have a pure-color frame in a video.
        video_padding_mask = clip_videos.reshape(batch_size, frames_per_video, -1).abs().mean(dim=-1) == 0.0
        video_padding_mask = video_padding_mask.unsqueeze(-1).expand(batch_size, frames_per_video, image_num_patches)
        video_padding_mask = video_padding_mask.reshape(clip_videos.size(0), video_num_patches)
        image_position_idx = (
            torch.arange(w).unsqueeze(0).expand(h, w)
            + torch.arange(h).unsqueeze(1) * image_resnet_adaptor.cfg.image_bucket_size
            + 1
        )
        image_position_idx = image_position_idx.view(-1).to(device)
        image_position_ids = image_position_idx[None, :].expand(clip_videos.size(0), image_num_patches)
        frame_position_idx = torch.arange(frames_per_video).to(device) + 1
        frame_position_ids = frame_position_idx[None, :].expand(clip_videos.size(0), frames_per_video)

        image_pos_embed: torch.Tensor = image_resnet_adaptor.embed_image_positions(image_position_ids)
        frame_pos_embed: torch.Tensor = self.embed_frame_positions(frame_position_ids)
        video_pos_embed = image_pos_embed.unsqueeze(1) + frame_pos_embed.unsqueeze(2)

        video_embed = image_embed.reshape(batch_size, video_num_patches, 1024)  # num_features)

        video_pos_embed = video_pos_embed.reshape(
            batch_size, video_num_patches, video_pos_embed.size(-1)
        )  # num_features)

        return video_embed, video_num_patches, video_padding_mask, image_position_ids, video_pos_embed

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot): ModalityType.VIDEO
        Returns:
            AdaptorOutput:
                - **embed** (Tensor): the processed embedding for OFA of
                  shape ``(src_len, batch, embed_dim)``
                - **padding_masks** (ByteTensor): the positions of
                  padding elements of shape ``(batch, src_len)``
                - **pos_embedding** (Tensor): the position embeddings
                  of shape ``(batch, src_len, embed_dim)``
                - **self_attn_bias** (List[Tensor]): attention bias in self attention
                  of shape ``(batch, num_attention_heads, src_len, src_len)``.
        """
        assert slot.modality == ModalityType.VIDEO
        image_resnet_adaptor = self.get_image_resnet_adaptor()
        (
            video_embed,
            video_num_patches,
            video_padding_mask,
            image_position_ids,
            video_pos_embed,
        ) = self.get_clip_videos_info(slot.value)
        video_embed = image_resnet_adaptor.image_proj(video_embed)

        batch_size, seq_length = video_embed.size()[:2]
        token_per_image = image_position_ids.size(-1)
        frame_count = seq_length // token_per_image
        self_attn_bias = []
        if self.cfg.use_self_attn_bias:
            for idx in range(self.num_layers):
                values_image = image_resnet_adaptor.get_rel_pos_bias(
                    batch_size, token_per_image, idx, image_position_ids=image_position_ids
                )
                values_frame = (
                    self.get_rel_pos_bias(batch_size, frame_count, idx).transpose(1, 2).transpose(0, 1).contiguous()
                )
                values_image = values_image.view(
                    values_image.size(0), values_image.size(1), 1, values_image.size(2), 1, values_image.size(3)
                )
                values_frame = values_frame.view(
                    values_frame.size(0), values_frame.size(1), 1, values_frame.size(1), 1
                )
                values = values_frame + values_image
                values = values.view(
                    values.size(0), values.size(1), values.size(2) * values.size(3), values.size(4) * values.size(5)
                )
                self_attn_bias.append(values)
        else:
            self_attn_bias = [None] * self.num_layers

        return AdaptorOutput(video_embed, video_padding_mask, video_pos_embed, self_attn_bias)

    def upgrade_state_dict_named(self, state_dict, name):
        if name == 'encoder.adaptor.video_image_sequence':
            resnet_prefix = name.replace('video_image_sequence', 'image_resnet')
            keys = [
                'layernorm_embedding.weight', 'layernorm_embedding.bias',
                'layernorm_position.weight', 'layernorm_position.bias',
                'type_embedding.weight',
            ]
            for key in keys:
                full_key = f'{name}.{key}'
                if full_key not in state_dict:
                    state_dict[full_key] = state_dict[f'{resnet_prefix}.{key}'].clone()
