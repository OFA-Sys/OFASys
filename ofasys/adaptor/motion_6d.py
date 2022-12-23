# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

from ofasys import ModalityType
from ofasys.adaptor.base import AdaptorOutput, Slot
from ofasys.adaptor.text import TextAdaptor, TextAdaptorConfig
from ofasys.configure import register_config
from ofasys.module import Embedding, utils
from ofasys.preprocessor import Dictionary

logger = logging.getLogger(__name__)


@dataclass
class Motion6dAdaptorConfig(TextAdaptorConfig):
    max_data_dim: int = field(default=512, metadata={"help": ""})
    max_noise_levels: int = field(default=1024, metadata={"help": ""})


class InputPadding(nn.Module):
    def __init__(self, dim):
        super(InputPadding, self).__init__()
        self.dim = dim

    def forward(self, x):
        batch_size, num_frames, dim = x.shape
        assert dim <= self.dim
        zeros = torch.zeros(batch_size, num_frames, self.dim - dim, dtype=x.dtype, device=x.device)
        x = torch.cat([x, zeros], dim=-1)
        assert x.shape == (batch_size, num_frames, self.dim)
        return x


@register_config("ofasys.adaptor", "motion_6d", Motion6dAdaptorConfig)
class Motion6dAdaptor(TextAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: Motion6dAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)
        self.input_padding = InputPadding(cfg.max_data_dim)
        self.frame_encoder = nn.Sequential(
            nn.Linear(cfg.max_data_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.frame_decoder = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.max_data_dim),
        )
        self.noise_level_emb = nn.Embedding(cfg.max_noise_levels, cfg.embed_dim * 2)
        self.noise_level_emb.weight.data.zero_()

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot): ModalityType.Motion

        Returns:
            AdaptorOutput:
                - **embed** (Tensor): the processed embedding for OFA of
                  shape ``(src_len, batch, embed_dim)``
                - **padding_masks** (ByteTensor): the positions of
                  padding elements of shape ``(batch, src_len)``
                - **pos_embedding** (Tensor): the position embeddings
                  of shape ``(batch, src_len, embed_dim)``

        """
        assert slot.modality in (ModalityType.MOTION,)
        input_value = slot.value['value']
        batch_size, seq_len, data_dim = input_value.shape

        padding_masks = slot.value['masks']
        assert padding_masks.shape == (batch_size, seq_len)

        known_w = slot.value.get('known_w', None)
        if known_w is not None:
            known_v = slot.value['value_0']
            assert known_w.shape == (batch_size, seq_len)
            assert known_v.shape == (batch_size, seq_len, data_dim)
            input_value = known_w.unsqueeze(-1) * known_v + (1.0 - known_w.unsqueeze(-1)) * input_value

        token_embed = self.frame_encoder(self.input_padding(input_value))
        embed_dim = token_embed.shape[-1]

        noise_level = slot.value.get('noise_level', None)
        if noise_level is not None:
            noise_level = (noise_level + 1).unsqueeze(-1)
            assert noise_level.shape == (batch_size, 1)
            if known_w is not None:
                noise_level = (known_w < 0.5).to(dtype=noise_level.dtype) * noise_level  # [B,T]
            scale, shift = self.noise_level_emb(noise_level).chunk(chunks=2, dim=-1)
            token_embed = (scale + 1.0) * token_embed + shift

        assert token_embed.shape == (batch_size, seq_len, embed_dim)
        pos_embed = self.embed_positions(utils.new_arange(token_embed, *token_embed.shape[:-1]))
        return AdaptorOutput(token_embed, padding_masks, pos_embed, [])

    def forward_output(self, x: Tensor, extra: Dict[str, Any], slot: Slot, **kwargs):
        """
        Args:
            x (Tensor): hidden states from model in the shape of
             ``(batch_size, seq_length, embed_dim)``
            extra (Dict[str, Any]): extra model output information.
            slot (Slot):  input preprocessed data.

        Returns:
            tuple:
                - x (Tensor): Tensor of shape ``(batch_size, seq_len, data_dim)``.
                - extra (Dict[str, Any]): model output with any modality-specific information.
        """
        batch_size, seq_len, data_dim = slot.value['value'].shape
        out = self.frame_decoder(x)[:, :, :data_dim]
        assert out.shape == (batch_size, seq_len, data_dim)
        return out, extra
