# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ofasys.adaptor import AdaptorOutput, OFAGeneralAdaptor
from ofasys.distributed import fsdp_wrap
from ofasys.module import (
    AdaptiveSoftmax,
    BaseLayer,
    LayerDropModuleList,
    LayerNorm,
    Linear,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    checkpoint_wrapper,
    utils,
)
from ofasys.preprocessor import Dictionary, Slot

from .base_encoder import BaseEncoder
from .incremental_decoder import IncrementalDecoder

logger = logging.getLogger(__name__)


class TransformerEncoder(BaseEncoder):
    """
    Transformer encoder consisting of *cfg.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        cfg (GeneralistModelConfig): parsed command-line arguments
        dictionary (Dictionary): global dictionary
    """

    def __init__(self, cfg, dictionary: Dictionary):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        OFAGeneralAdaptor._embed_tokens = None  # rm the existing embed to build a new embed
        self.adaptor = OFAGeneralAdaptor(cfg, dictionary, True)
        if cfg.checkpoint_adaptor_activations:
            self.adaptor = checkpoint_wrapper(self.adaptor, cfg.offload_activations)

        if cfg.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=cfg.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = torch.linspace(0, cfg.encode_drop_path_rate, cfg.encoder_layers)
        self.layers.extend([self.build_encoder_layer(cfg, drop_path_rate=dpr[i]) for i in range(cfg.encoder_layers)])

        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg, drop_path_rate=0.0):
        layer = TransformerEncoderLayer(cfg, drop_path_rate=drop_path_rate)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(self, slots: List[Slot], return_all_hiddens: bool = False, return_all_attention_weights: bool = False):
        """
        Args:
            slots (List[Slot]): preprocessed data
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            return_all_attention_weights (bool, optional): also return all attention weights (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape ``(src_len, batch, embed_dim)``
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape ``(batch, src_len)``
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape ``(batch, src_len, embed_dim)``
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape ``(src_len, batch, embed_dim)``.
                  Only populated if *return_all_hiddens* is True.
                - **position_embeddings** (Tensor): the position embedding lookup
                  of shape ``(batch, src_len, embed_dim)``
                  - **encoder_attention_weights** (Tensor): attention weights of encoder's self attention of
                  shape ``(num_heads, batch_size, src_len, src_len)``.
                  Only return if *return_all_attention_weights* and *return_encoder_out* are both True.
        """
        if len(slots) == 0:
            return None
        ret = self.adaptor(slots)
        adaptor_output = AdaptorOutput(*ret)

        # B x T x C -> T x B x C
        x = adaptor_output.embed.transpose(0, 1)
        has_pad = adaptor_output.masks.any()
        if has_pad:
            adaptor_output.embed *= 1 - adaptor_output.masks.unsqueeze(-1).type_as(adaptor_output.embed)

        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)

        encoder_attention_states = []
        # encoder layers
        for idx, layer in enumerate(self.layers):
            if self.cfg.use_self_attn_bias:
                if self.cfg.share_attn_bias:
                    self_attn_bias = adaptor_output.self_attn_bias[0]
                else:
                    self_attn_bias = adaptor_output.self_attn_bias[idx]
                self_attn_bias = self_attn_bias.view(-1, x.size(0), x.size(0))
            else:
                self_attn_bias = None
            x, self_attn_weights = layer(
                x,
                encoder_padding_mask=adaptor_output.masks if has_pad else None,
                self_attn_bias=self_attn_bias,
                need_attn=return_all_attention_weights,
                modal_mask=adaptor_output.modal_mask,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            if return_all_attention_weights:
                encoder_attention_states.append(self_attn_weights)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [adaptor_output.masks],  # B x T
            "encoder_embedding": [adaptor_output.embed],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "position_embeddings": [adaptor_output.pos_embed],  # B x T x C
            "encoder_attention_weights": encoder_attention_states,
        }

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [encoder_out["encoder_padding_mask"][0].index_select(0, new_order)]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [encoder_out["encoder_embedding"][0].index_select(0, new_order)]
        if len(encoder_out["position_embeddings"]) == 0:
            new_position_embeddings = []
        else:
            new_position_embeddings = [(encoder_out["position_embeddings"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "position_embeddings": new_position_embeddings,  # B x T x C
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return self.max_source_positions


class TransformerDecoder(IncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        cfg (GeneralistModelConfig): arguments
        dictionary (Dictionary): decoding dictionary
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        no_encoder_attn=False,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.adaptor = OFAGeneralAdaptor(cfg, dictionary, False)
        if cfg.checkpoint_adaptor_activations:
            self.adaptor = checkpoint_wrapper(self.adaptor, cfg.offload_activations)

        self.share_input_output_embed = cfg.share_decoder_input_output_embed
        self.num_attention_heads = cfg.decoder_attention_heads

        embed_dim = cfg.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = int(cfg.decoder_output_dim)
        if self.cfg.use_self_attn_bias:
            self.cross_pos_q_linear = nn.Linear(embed_dim, embed_dim)
            self.cross_pos_k_linear = nn.Linear(embed_dim, embed_dim)
        self.cross_self_attention = cfg.cross_self_attention

        if cfg.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=cfg.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = torch.linspace(0, cfg.encode_drop_path_rate, cfg.encoder_layers)
        self.layers.extend(
            [self.build_decoder_layer(cfg, no_encoder_attn, drop_path_rate=dpr[i]) for i in range(cfg.decoder_layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None

    def build_decoder_layer(self, cfg, no_encoder_attn=False, drop_path_rate=0.0):
        layer = TransformerDecoderLayer(cfg, no_encoder_attn, drop_path_rate=drop_path_rate)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_cross_pos_info(self, embed, tgt_pos_embed, src_pos_embed):
        """
        Compute abs position bias for cross attention.
        """
        batch_size = embed.size(0)
        tgt_len = embed.size(1)
        src_len = src_pos_embed.size(1)
        pos_q = (
            self.cross_pos_q_linear(tgt_pos_embed)
            .view(batch_size, tgt_len, self.num_attention_heads, -1)
            .transpose(1, 2)
            * self.adaptor.pos_scaling
        )
        pos_k = (
            self.cross_pos_k_linear(src_pos_embed)
            .view(batch_size, src_len, self.num_attention_heads, -1)
            .transpose(1, 2)
        )
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))
        return abs_pos_bias

    def forward(
        self,
        slots: List[Slot],
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
        return_all_attention_weights: bool = False,
    ):
        """
        Args:
            slots (List[Slot]): preprocessed data
            encoder_out (optional, Dict[str, List[Tensor]]): output from the encoder,
                used for encoder-side attention.
            incremental_state (dict): dictionary used for storing state during
                Incremental decoding
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            return_all_attention_weights (bool, optional): also return all attention weights (default: False).

        Returns:
            tuple:
                - the decoder's output: the decoder's features of shape ``(batch, tgt_len, embed_dim)``
                  if *features_only* is True, else return outputs from adaptor.
                - a dictionary with decoder extra outputs.

                    * **attn** (List[Tensor]) : return specific attention weights
                    * **inner_states** (List[Tensor]): all intermediate encoder hidden states
                      of shape ``(tgt_len, batch, embed_dim)``,
                    * **decoder_attentions** (List[Tensor]): attention weights of decoder's self attention of
                      shape ``(num_heads, batch_size, tgt_len, tgt_len)``.
                      Only return if *return_all_attention_weights* is True.
                    * **cross_attentions** (List[Tensor]): attention weights of decoder's self attention of
                      shape ``(num_heads, batch_size, src_len, tgt_len)``.
                      Only return if *return_all_attention_weights* is True.
        """

        x, extra = self.extract_features(
            slots,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            return_all_hiddens=return_all_hiddens,
            return_all_attention_weights=return_all_attention_weights,
        )
        extra['last_hidden_state'] = x
        if not features_only:
            adaptor_output, extra = self.adaptor.forward_output(x, extra, slots)
            return adaptor_output, extra
        return x, extra

    def extract_features(
        self,
        slots: List[Slot],
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
        return_all_attention_weights: bool = False,
    ):

        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            slots (List[Slot]): preprocessed data.
            encoder_out (optional, Dict[str, List[Tensor]]): output from the encoder,
                used for encoder-side attention.
            incremental_state (dict): dictionary used for storing state during Incremental decoding.
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            return_all_attention_weights (bool, optional): also return all attention weights (default: False).

        Returns:
            tuple:
                - the decoder's features of shape ``(batch, tgt_len, embed_dim)``.
                - a dictionary with decoder extra outputs.

                    * **attn** (List[Tensor]) : return specific attention weights.
                    * **inner_states** (List[Tensor]): all intermediate encoder hidden states of
                      shape ``(tgt_len, batch, embed_dim)``.
                    * **decoder_attentions** (List[Tensor]): attention weights of decoder's self attention of
                      shape ``(num_heads, batch_size, tgt_len, tgt_len)``.
                      Only return if *return_all_attention_weights* is True.
                    * **cross_attentions** (List[Tensor]): attention weights of decoder's self attention of
                      shape ``(num_heads, batch_size, src_len, tgt_len)``.
                      Only return if *return_all_attention_weights* is True.


        """

        ret = self.adaptor(slots)
        adaptor_output = AdaptorOutput(*ret)

        bsz, slen = adaptor_output.embed.size()[:2]

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        src_pos_embed: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert enc.size()[1] == bsz, f"Expected enc.shape == (t, {bsz}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        if encoder_out is not None and len(encoder_out["position_embeddings"]) > 0:
            src_pos_embed = encoder_out['position_embeddings'][0]

        tgt_embed = adaptor_output.embed
        tgt_pos_embed = adaptor_output.pos_embed
        self_attn_padding_mask = adaptor_output.masks
        all_self_attn_bias = adaptor_output.self_attn_bias

        # TODO: better arg
        if not self.cfg.entangle_position_embedding:
            cross_abs_pos_bias = self.get_cross_pos_info(tgt_embed, tgt_pos_embed, src_pos_embed=src_pos_embed)
            cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])
        else:
            cross_abs_pos_bias = None

        if incremental_state is not None:
            tgt_embed = tgt_embed[:, -1:]
            cross_abs_pos_bias = cross_abs_pos_bias[:, -1:, :] if cross_abs_pos_bias is not None else None
            self_attn_padding_mask = self_attn_padding_mask[:, -1:]

        # B x T x C -> T x B x C
        x = tgt_embed.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = []
        decoder_attentions: List[Optional[Tensor]] = []
        cross_attentions: List[Optional[Tensor]] = []

        if return_all_hiddens:
            inner_states.append(x)
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            if self.cfg.use_self_attn_bias:
                if self.cfg.share_attn_bias:
                    self_attn_bias = all_self_attn_bias[0]
                else:
                    self_attn_bias = all_self_attn_bias[idx]
                self_attn_bias = self_attn_bias.view(-1, *self_attn_bias.size()[-2:])
                if incremental_state is not None:
                    self_attn_bias = self_attn_bias[:, -1:, :]
            else:
                self_attn_bias = False

            x, layer_self_attn, layer_cross_attn = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer) or return_all_attention_weights),
                need_head_weights=bool((idx == alignment_layer)),
                self_attn_bias=self_attn_bias,
                cross_attn_bias=cross_abs_pos_bias,
                modal_mask=adaptor_output.modal_mask,
            )

            if return_all_attention_weights:
                decoder_attentions.append(layer_self_attn)
                cross_attentions.append(layer_cross_attn)

            inner_states.append(x)
            if layer_self_attn is not None and idx == alignment_layer:
                attn = layer_self_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {
            "attn": [attn],
            "inner_states": inner_states,
            "decoder_attentions": decoder_attentions,
            "cross_attentions": cross_attentions,
        }

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.cfg.max_target_positions

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript.
        # This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
