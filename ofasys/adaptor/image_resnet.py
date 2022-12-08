# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ofasys import ModalityType
from ofasys.adaptor.base import AdaptorOutput, BaseAdaptor, BaseAdaptorConfig, Slot
from ofasys.configure import ChoiceEnum, register_config
from ofasys.module import resnet50_backbone  # noqa
from ofasys.module import resnet101_backbone  # noqa
from ofasys.module import resnet152_backbone  # noqa
from ofasys.module import Embedding, Linear, SynBatchNorm2d
from ofasys.preprocessor import Dictionary
from ofasys.utils.file_utils import cached_path

logger = logging.getLogger(__name__)


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
class ImageResnetAdaptorConfig(BaseAdaptorConfig):
    resnet_type: ChoiceEnum(['resnet50', 'resnet101', 'resnet152']) = field(
        default='resnet152',
        metadata={"help": "resnet type"},
    )
    resnet_drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "resnet drop path rate"},
    )
    sync_bn: bool = field(
        default=False,
        metadata={"help": "sync batchnorm"},
    )
    freeze_resnet: bool = field(
        default=False,
        metadata={"help": "freeze resnet"},
    )
    image_bucket_size: int = field(
        default=42,
        metadata={"help": "image bucket size"},
    )
    pretrained_ckpt_path: str = field(default="", metadata={"help": "path of pretrained ckpt"})


@register_config("ofasys.adaptor", "image_resnet", ImageResnetAdaptorConfig)
class ImageResnetAdaptor(BaseAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: ImageResnetAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)

        self.embed_image_positions = Embedding(cfg.image_bucket_size**2 + 1, cfg.embed_dim)

        resnet_backbone = {
            'resnet50': resnet50_backbone,
            'resnet101': resnet101_backbone,
            'resnet152': resnet152_backbone,
        }[cfg.resnet_type]
        self.embed_images = resnet_backbone(
            norm_layer=SynBatchNorm2d if cfg.sync_bn else None,
            drop_path_rate=cfg.resnet_drop_path_rate,
        )
        self.image_proj = Linear(1024, cfg.embed_dim)
        if self.cfg.pretrained_ckpt_path:
            local_model_path = cached_path(self.cfg.pretrained_ckpt_path)
            sd = torch.load(local_model_path, map_location="cpu")
            logger.info(
                f'loading adaptor ckpt from {self.cfg.pretrained_ckpt_path}, {self.embed_images.load_state_dict(sd)}'
            )

        image_num_rel_dis = (2 * cfg.image_bucket_size - 1) * (2 * cfg.image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(cfg.image_bucket_size, image_num_rel_dis)
        num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
        self.image_rel_pos_table_list = nn.ModuleList(
            [Embedding(image_num_rel_dis, cfg.num_attention_heads, zero_init=True) for _ in range(num_rel_pos_tables)]
        )
        self.register_buffer("image_rp_bucket", image_rp_bucket)

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_resnet:
            for m in self.embed_images.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def get_rel_pos_bias(self, batch_size, seq_length, idx, **kwargs):
        image_position_ids = kwargs.get('image_position_ids')
        rp_bucket_size = self.image_rp_bucket.size(1)
        rp_bucket = (
            self.image_rp_bucket.unsqueeze(0)
            .expand(batch_size, rp_bucket_size, rp_bucket_size)
            .gather(1, image_position_ids[:, :, None].expand(batch_size, seq_length, rp_bucket_size))
            .gather(2, image_position_ids[:, None, :].expand(batch_size, seq_length, seq_length))
        )
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(0, 3, 1, 2)

        return values

    def get_patch_images_info(self, patch_images):
        """
        Args:
            patch_images (Tensor): stacked image tensors of size ``(batch_size, num_channels, height, width)``

        Returns:
            Tuple:
                - **image_embed** (Tensor): the processed embedding for OFA of
                  shape ``(src_len, batch, embed_dim)``
                - **image_num_patches** (int): the number of image patches.
                - **image_padding_masks** (ByteTensor): the positions of
                  padding elements of shape ``(batch, src_len)``
                - **image_position_ids** (Tensor): the position ids
                  of shape ``(batch, src_len)``
                - **image_pos_embed** (Tensor): the position embeddings
                  of shape ``(batch, src_len, embed_dim)``

        """
        device = patch_images.device
        image_embed = self.embed_images(patch_images)
        h, w = image_embed.shape[-2:]
        image_num_patches = h * w
        image_padding_mask = torch.zeros((patch_images.size(0), image_num_patches), dtype=torch.bool, device=device)
        image_position_idx = (
            torch.arange(w, device=device).unsqueeze(0).expand(h, w)
            + torch.arange(h, device=device).unsqueeze(1) * self.cfg.image_bucket_size
            + 1
        )
        image_position_idx = image_position_idx.view(-1)
        image_position_ids = image_position_idx[None, :].expand(patch_images.size(0), image_num_patches)

        image_embed = image_embed.flatten(2).transpose(1, 2)
        image_pos_embed = self.embed_image_positions(image_position_ids)

        return image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot):
                - ModalityType: IMAGE
                - value: stacked image tensors of size ``(batch_size, num_channels, height, width)``

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
        assert slot.modality == ModalityType.IMAGE
        (
            image_embed,
            image_num_patches,
            image_padding_mask,
            image_position_ids,
            image_pos_embed,
        ) = self.get_patch_images_info(slot.value)
        image_embed = self.image_proj(image_embed)

        batch_size, seq_length = image_embed.size()[:2]
        self_attn_bias = []
        if self.cfg.use_self_attn_bias:
            num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
            for idx, layer in enumerate(range(num_rel_pos_tables)):
                values = self.get_rel_pos_bias(batch_size, seq_length, idx, image_position_ids=image_position_ids)
                self_attn_bias.append(values)

        return AdaptorOutput(image_embed, image_padding_mask, image_pos_embed, self_attn_bias)
