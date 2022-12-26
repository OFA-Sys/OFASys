# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from ofasys.configure import BaseDataclass
from ofasys.module import Dropout, Embedding, LayerNorm
from ofasys.preprocessor import Dictionary, Slot


@dataclass
class AdaptorOutput:
    """
    Args:
        embed (torch.FloatTensor): the processed embedding for OFA of shape ``(batch_size, seq_length, hidden_size)``
        masks (torch.BoolTensor):  the positions of
                  padding elements of shape ``(batch, src_len)``
        pos_embed (torch.FloatTensor): the position embeddings of shape ``(batch_size, seq_length, hidden_size)``
        self_attn_bias (List[torch.FloatTensor], optional): attention bias in self attention
                  of shape ``(batch_size, num_attention_heads, seq_length, seq_length)``

    """

    # B: batch_size
    # T: seq_length
    # H: hidden_size
    # A: num_attention_heads

    embed: torch.FloatTensor  # B x T x H
    masks: torch.BoolTensor  # B x T
    pos_embed: torch.FloatTensor  # B x T x H
    self_attn_bias: List[torch.FloatTensor]  # List[B x A x T x T]
    modal_mask: torch.IntTensor = None  # B x T

    def __post_init__(self):
        assert self.embed is not None
        batch_size, seq_length, hidden_size = self.embed.shape
        if self.masks is not None:
            assert self.masks.shape == (batch_size, seq_length)
        if self.pos_embed is not None:
            assert self.pos_embed.shape == (batch_size, seq_length, hidden_size)

    @property
    def seq_length(self):
        return self.embed.shape[1]


@dataclass
class BaseAdaptorConfig(BaseDataclass):
    is_active: bool = field(default=False, metadata={"help": "is active for config_store"})
    layernorm_embedding: bool = field(default=True, metadata={"help": "add layernorm to embedding"})
    layernorm_position: bool = field(default=True, metadata={"help": "add layernorm to position emb"})
    add_type_embedding: bool = field(
        default=True,
        metadata={"help": "add source/region/patch type embedding"},
    )
    entangle_position_embedding: bool = field(
        default=False,
        metadata={"help": "entangle position embedding"},
    )
    no_scale_embedding: bool = field(default=True, metadata={"help": "if True, do not scale embeddings"})
    scale_embedding_gradient: float = field(
        default=1.0,
        metadata={"help": "scale embedding gradient after adaptor backbone"},
    )
    dropout: float = None
    embed_dim: int = None
    num_attention_heads: int = None
    encoder_layers: int = None
    decoder_layers: int = None
    max_position: int = None
    use_self_attn_bias: bool = None
    share_attn_bias: bool = None

    def parse_from_model_cfg(self, model_cfg):
        # TODO: change it to II
        self.dropout = model_cfg.dropout if self.dropout is None else self.dropout
        self.embed_dim = model_cfg.encoder.embed_dim if self.embed_dim is None else self.embed_dim
        self.num_attention_heads = (
            model_cfg.encoder.attention_heads if self.num_attention_heads is None else self.num_attention_heads
        )
        self.encoder_layers = model_cfg.encoder.layers if self.encoder_layers is None else self.encoder_layers
        self.decoder_layers = model_cfg.decoder.layers if self.decoder_layers is None else self.decoder_layers
        self.max_position = model_cfg.max_source_positions if self.max_position is None else self.max_position
        self.use_self_attn_bias = (
            model_cfg.use_self_attn_bias if self.use_self_attn_bias is None else self.use_self_attn_bias
        )
        self.share_attn_bias = model_cfg.share_attn_bias if self.share_attn_bias is None else self.share_attn_bias
        self.entangle_position_embedding = (
            model_cfg.entangle_position_embedding
            if self.entangle_position_embedding is None
            else self.entangle_position_embedding
        )


class BaseAdaptor(torch.nn.Module):
    def __init__(
            self,
            embed_tokens: Embedding,
            dictionary: Dictionary,
            is_src: bool,
            general_adaptor,
            cfg: BaseAdaptorConfig,
    ):
        """
        IO Adaptors convert modality data between its tensor form that is represented by computer program and
        embedding sequence that is readable by the universal computation model, e.g. OFA.
        In order to keep the Input Adaptors and the Output Adaptors are used in pairs, we use two methods in the
        same class to represent a pair of IO adaptors (**forward** for Input Adaptor and **forward_output**
        for Output Adaptor).

        Args:
            embed_tokens (Embedding): global embedding matrix.
            dictionary (Dictionary): global vocab.
            is_src (bool): where is the adaptor used for .
            general_adaptor (GeneralAdaptor): instance of GeneralAdaptor.
            cfg (BaseAdaptorConfig): adaptor config.
        """
        super().__init__()
        # "self.embed_tokens = embed_tokens" will register it as a child module, which may
        # cause some troubles. We instead use lambda to NOT register it as a child module.
        self.embed_tokens = lambda x: embed_tokens(x)
        self.embed_tokens_T = lambda x: F.linear(x, embed_tokens.weight)
        self.dictionary = dictionary
        self.is_src = is_src
        # "self.general_adaptor = general_adaptor" will register it as a child module,
        # which cause to call the function "move_to_cuda" recursively.
        self._general_adaptor = [general_adaptor]
        self.cfg = cfg
        self.num_layers = cfg.encoder_layers if is_src else cfg.decoder_layers

        self.dropout_module = Dropout(cfg.dropout, module_name=self.__class__.__name__)
        self.layernorm_embedding = LayerNorm(cfg.embed_dim) if cfg.layernorm_embedding else None
        self.layernorm_position = LayerNorm(cfg.embed_dim) if cfg.layernorm_position else None
        self.type_embedding = Embedding(1, cfg.embed_dim) if cfg.add_type_embedding else None
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(cfg.embed_dim)

        self.register_forward_hook(BaseAdaptor.forward_hook_fn)

    @property
    def general_adaptor(self):
        return self._general_adaptor[0]

    def forward_hook_fn(self, inputs, output: AdaptorOutput):
        """
        This hook will be called every time after :func:`forward` has computed an output.
        Position embedding, type_embedding, layernorm_embedding and layernorm_position will be added to adaptor output.
        If output does not contain self_attn_bias, this hook will generate self_attn_bias list by calling
        ``get_rel_pos_bias()`` and expand them according to batch size.

        Args:
            inputs: model input.
            output: AdaptorOutput computed by the adaptor.

        Returns:
            AdaptorOutput:
                modified adaptor_output
        """
        slot: Slot = inputs[0]
        embed = self.embed_scale * output.embed

        if self.cfg.entangle_position_embedding and output.pos_embed is not None:
            embed += output.pos_embed
        if slot.is_src and self.type_embedding is not None:
            embed += self.type_embedding.weight.squeeze()
        if self.cfg.scale_embedding_gradient != 1.0:
            alpha = self.cfg.scale_embedding_gradient
            embed = embed * alpha + embed.detach() * (1 - alpha)
        if self.layernorm_embedding is not None:
            embed = self.layernorm_embedding(embed)
        if self.layernorm_position is not None and output.pos_embed is not None:
            output.pos_embed = self.layernorm_position(output.pos_embed)
        output.embed = self.dropout_module(embed)

        if not output.self_attn_bias and self.cfg.use_self_attn_bias:
            output.self_attn_bias = []
            batch_size, seq_length = output.embed.size()[:2]
            num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
            for idx in range(num_rel_pos_tables):
                values = self.get_rel_pos_bias(batch_size, seq_length, idx)
                output.self_attn_bias.append(self.expand_rel_pos_bias(values, batch_size))

        return output

    @abstractmethod
    def forward(self, inputs: Union[Slot, List[Slot]], **kwargs) -> AdaptorOutput:
        """
        The Adaptor work as the InputAdaptor, takes corresponding data in tensor format as input,
        and then output sequences in the same format -ref **AdaptorOutput**.

        Args:
            inputs (Slot): preprocessed input data.

        Returns:
            AdaptorOutput:
                adaptor_output: adaptor output for the input slot.
        """
        raise NotImplementedError

    def forward_output(self, x: Tensor, extra: Dict[str, Any], slot: Slot, **kwargs):
        """
        The Adaptor work as the OutputAdaptor, takes hidden states from model as input,
        and then output the modality data in their own form, e.g. probs on vocabulary.

        Args:
            x (Tensor): hidden states from model in the shape of
             ``(batch_size, seq_length, embed_dim)``
            extra (Dict[str, Any]): extra model output information.
            slot (Slot):  input preprocessed data.

        Returns:
            tuple:
                - x (Tensor): modality data in Tensor form.
                - extra (Dict[str, Any]): model output with any modality-specific information.
        """
        return x, extra

    @abstractmethod
    def get_rel_pos_bias(self, batch_size, seq_length, idx, **kwargs):
        """
        Get relative position bias of self attention.

        Args:
            batch_size: batch size of input data.
            seq_length: sequence length of input data.
            idx: layer index.

        Returns:
            Tensor:
                attention bias.
        """
        raise NotImplementedError

    def expand_rel_pos_bias(self, values: Tensor, batch_size: int):
        """
        Expand and permute attention bias.

        Args:
            values (Tensor): origin self attention bias of shape ``(seq_length, seq_length, num_attention_heads)``.
            batch_size (Int): batch size of input data.

        Returns:
            Tensor:
                expanded attention bias of shape ``(batch_size, num_attention_heads, seq_length, seq_length)``
        """
        values = values.unsqueeze(0).expand(batch_size, -1, -1, -1)
        values = values.permute([0, 3, 1, 2])
        return values

    def upgrade_state_dict_named(self, state_dict, name):
        pass

    def update_sample(self, sample):
        # use for some of the input need to be processed on GPU
        return sample

    def check_adaptor_slot(self, slot):
        return self.general_adaptor.get_adaptor(slot) is self
