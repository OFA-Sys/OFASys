# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ofasys.adaptor.base import AdaptorOutput, BaseAdaptor, BaseAdaptorConfig, Slot
from ofasys.configure import register_config
from ofasys.module import Embedding, utils
from ofasys.preprocessor import Dictionary


def make_token_bucket_position(bucket_size, max_position):
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


@dataclass
class TextAdaptorConfig(BaseAdaptorConfig):
    token_bucket_size: int = field(
        default=256,
        metadata={"help": "token bucket size"},
    )
    share_input_output_embed: bool = field(
        default=True,
        metadata={"help": "share_input_output_embed"},
    )
    output_embed_dim: Optional[int] = field(
        default=512,
        metadata={"help": "output_dim"},
    )
    output_dim: Optional[int] = field(
        default=None,
        metadata={"help": "output_dim"},
    )
    output_bias: bool = field(
        default=False,
        metadata={"help": "output_embed_bias"},
    )


@register_config("ofasys.adaptor", "text", TextAdaptorConfig)
class TextAdaptor(BaseAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: TextAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)

        self.embed_positions = Embedding(cfg.max_position + 2, cfg.embed_dim)

        token_num_rel_dis = 2 * cfg.token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(cfg.token_bucket_size, cfg.max_position)

        num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
        self.token_rel_pos_table_list = nn.ModuleList(
            [Embedding(token_num_rel_dis, cfg.num_attention_heads, zero_init=True) for _ in range(num_rel_pos_tables)]
        )

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        # TODO: use II("model.share_all_embeddings") when II is supported
        self.share_input_output_embed = True
        if not cfg.share_input_output_embed:
            self.share_input_output_embed = False
        self.output_dim: int = cfg.output_dim
        if self.output_dim is None:
            self.output_dim = len(dictionary)

        self.output_embed_dim = cfg.output_embed_dim

        self.output_embed_bias: bool = cfg.output_bias
        self.output_projection = None
        self.build_output_projection(dictionary)

    def build_output_projection(self, dictionary):
        if self.share_input_output_embed:
            self.output_projection = self.embed_tokens_T
        else:
            self.output_projection = nn.Linear(self.output_embed_dim, self.output_dim, bias=self.output_embed_bias)
            nn.init.normal_(self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5)

    def get_rel_pos_bias(self, batch_size, seq_length, idx, **kwargs):
        rp_bucket = self.token_rp_bucket[:seq_length, :seq_length]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        return values

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot): ModalityType.Text
        Returns:
            AdaptorOutput:
                - **embed** (Tensor): the processed embedding for OFA of
                  shape ``(src_len, batch, embed_dim)``
                - **padding_masks** (ByteTensor): the positions of
                  padding elements of shape ``(batch, src_len)``
                - **pos_embedding** (Tensor): the position embeddings
                  of shape ``(batch, src_len, embed_dim)``
        """
        src_tokens = slot.value
        if self.dictionary.pad() is not None:
            padding_masks = src_tokens.eq(self.dictionary.pad())
        else:
            padding_masks = torch.zeros_like(src_tokens, dtype=torch.bool)
        pos_embed = self.embed_positions(utils.new_arange(src_tokens))
        token_embedding = self.embed_tokens(src_tokens)

        return AdaptorOutput(token_embedding, padding_masks, pos_embed, [])

    def forward_output(self, x: Tensor, extra: Dict[str, Any], slot: Slot, **kwargs):
        """
        Args:
            x (Tensor): hidden states from model in the shape of
             ``(batch_size, seq_length, embed_dim)``
            extra (Dict[str, Any]): extra model output information.
            slot (Slot):  input preprocessed data.

        Returns:
            tuple:
                - x (Tensor): Tensor of shape ``(batch_size, seq_length, vocab_size)``.
                - extra (Dict[str, Any]): model output with any modality-specific information.
        """
        return self.output_projection(x), extra
