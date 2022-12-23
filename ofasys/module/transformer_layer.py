# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ofasys import ModalityType
from ofasys.module import utils
from ofasys.module.sparse_dispatcher import SparseDispatcher
from . import Dropout, DropPath, LayerNorm, MultiheadAttention, TransformerConfig


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, drop_path_rate=0.0):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = Dropout(cfg.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = Dropout(float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = nn.Linear(self.embed_dim, cfg.encoder.ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.encoder.ffn_embed_dim, self.embed_dim)
        if cfg.modal_ffn:
            self.experts_num =  int(len(ModalityType))
            self.experts_fc1 = nn.ModuleList([copy.deepcopy(self.fc1) for i in range(self.experts_num)])
            self.experts_fc2 = nn.ModuleList([copy.deepcopy(self.fc2) for i in range(self.experts_num)])
            self.gates_way = torch.eye(self.experts_num)
        self.attn_ln = LayerNorm(self.embed_dim) if cfg.scale_attn else None
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        self.ffn_layernorm = LayerNorm(cfg.encoder_ffn_embed_dim) if cfg.scale_fc else None
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if cfg.scale_resids
            else None
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.drop_path = DropPath(drop_path_rate, batch_axis=1)

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            scale_factor=cfg.attn_scale_factor,
            scale_heads=cfg.scale_heads,
            use_fused=cfg.use_fused,
        )

    def residual_connection(self, x, residual):
        return residual + self.drop_path(x)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
                if "{}.{}.{}".format(name, new, m) not in state_dict and "{}.{}".format(new, m) in self.state_dict():
                    state_dict["{}.{}.{}".format(name, new, m)] = self.state_dict()["{}.{}".format(new, m)]

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                for i in range(self.experts_num):
                    if param_name.find("experts_fc1"+"."+str(i)) != -1:
                        state_dict[prefix + param_name] = self.state_dict()[param_name.replace("experts_fc1"+"."+str(i), "fc1")]
                    if param_name.find("experts_fc2"+"."+str(i)) != -1:
                        state_dict[prefix + param_name] = self.state_dict()[param_name.replace("experts_fc2"+"."+str(i), "fc2")]
                if param_name.find("experts_fc1") == -1 and param_name.find("experts_fc2") == -1:
                    state_dict[prefix + param_name] = self.state_dict()[param_name]

    def modal_for_ffn(self, modal_mask, x, experts_fc):
        modal_mask_fc = modal_mask
        bs, seq_len = modal_mask_fc.shape
        modal_mask_fc = modal_mask_fc.view(bs * seq_len)
        gates  = torch.index_select(self.gates_way.to(modal_mask_fc.device) , 0, modal_mask_fc)
        gates = gates.view(bs * seq_len, self.experts_num)
        dispatcher = SparseDispatcher(self.experts_num, gates)
        seq_len, bs, dim = x.shape
        x = x.view(seq_len * bs, dim)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [experts_fc[i](expert_inputs[i]) for i in range(self.experts_num)]
        x = dispatcher.combine(expert_outputs)
        x = x.view(seq_len, bs, -1)
        x = x.half()
        return x

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        self_attn_bias: Optional[Tensor] = None,
        need_attn: bool = False,
        modal_mask = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape ``(seq_len, batch, embed_dim)``
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                ``(batch, seq_len)`` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape ``(tgt_len, src_len)``,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
            self_attn_bias (Tensor):
            need_attn (bool):

        Returns:
            encoded output of shape ``(seq_len, batch, embed_dim)``

        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=need_attn,
            attn_mask=attn_mask,
            attn_bias=self_attn_bias,
        )
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        # ffn1 modal
        if self.cfg.modal_ffn:
            x = self.modal_for_ffn(modal_mask, x, self.experts_fc1)
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        # ffn2 modal
        if self.cfg.modal_ffn:
            x = self.modal_for_ffn(modal_mask, x, self.experts_fc2)
        else:
            x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, self_attn_weights


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        add_bias_kv (bool, optional): (default: False).
        add_zero_attn (bool, optional): (default: False).
        drop_path_rate (float, optional): (default: 0.0).
    """

    def __init__(
        self,
        args,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
        drop_path_rate=0.0,
    ):
        cfg = TransformerConfig.from_namespace(args)
        self.cfg = cfg
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = Dropout(cfg.dropout, module_name=self.__class__.__name__)

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.self_attn_ln = LayerNorm(self.embed_dim) if cfg.scale_attn else None
        self.cross_attn_ln = LayerNorm(self.embed_dim) if cfg.scale_attn else None
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = Dropout(float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = cfg.decoder.normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(cfg, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.ffn_layernorm = LayerNorm(cfg.decoder_ffn_embed_dim) if cfg.scale_fc else None
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if cfg.scale_resids
            else None
        )

        self.fc1 = nn.Linear(self.embed_dim, cfg.decoder.ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.decoder.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.drop_path = DropPath(drop_path_rate, batch_axis=1)

        if cfg.modal_ffn:
            self.experts_num =  int(len(ModalityType))
            self.experts_fc1 = nn.ModuleList([copy.deepcopy(self.fc1) for i in range(self.experts_num)])
            self.experts_fc2 = nn.ModuleList([copy.deepcopy(self.fc2) for i in range(self.experts_num)])
            self.gates_way = torch.eye(self.experts_num)

    def build_self_attention(self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            scale_factor=cfg.attn_scale_factor,
            scale_heads=cfg.scale_heads,
            use_fused=cfg.use_fused,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            scale_factor=cfg.attn_scale_factor,
            scale_heads=cfg.scale_heads,
            use_fused=cfg.use_fused,
        )

    def residual_connection(self, x, residual):
        return residual + self.drop_path(x)

    def modal_for_ffn(self, modal_mask, x, experts_fc):
        modal_mask_fc = modal_mask
        bs, seq_len = modal_mask_fc.shape
        modal_mask_fc = modal_mask_fc.view(bs * seq_len)
        gates  = torch.index_select(self.gates_way.to(modal_mask_fc.device) , 0, modal_mask_fc)
        gates = gates.view(bs * seq_len, self.experts_num)
        dispatcher = SparseDispatcher(self.experts_num, gates)
        seq_len, bs, dim = x.shape
        x = x.view(seq_len * bs, dim)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [experts_fc[i](expert_inputs[i]) for i in range(self.experts_num)]
        x = dispatcher.combine(expert_outputs)
        x = x.view(seq_len, bs, -1)
        x = x.half()
        return x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        self_attn_bias: Optional[Tensor] = None,
        cross_attn_bias: Optional[Tensor] = None,
        modal_mask = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape ``(seq_len, batch, embed_dim)``
            encoder_out:
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape ``(batch, src_len)`` where padding
                elements are indicated by ``1``.
            incremental_state:
            self_attn_mask:
            self_attn_padding_mask:
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
            self_attn_bias (Tensor, optional): attention bias for self attention.
            cross_attn_bias (Tensor, optional): attenion bias for cross attention.

        Returns:
            encoded output of shape ``(seq_len, batch, embed_dim)``
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat((x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(encoder_out.size(1), encoder_out.size(0))
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, self_attn_weights = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_attn,
            attn_mask=self_attn_mask,
            attn_bias=self_attn_bias,
        )
        if self.self_attn_ln is not None:
            x = self.self_attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        cross_attn_weights = None
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, cross_attn_weights = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                attn_bias=cross_attn_bias,
            )
            if self.cross_attn_ln is not None:
                x = self.cross_attn_ln(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # ffn1 modal
        if self.cfg.modal_ffn:
            x = self.modal_for_ffn(modal_mask[:x.shape[1],:x.shape[0]], x, self.experts_fc1)
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        # ffn2 modal
        if self.cfg.modal_ffn:
            x = self.modal_for_ffn(modal_mask[:x.shape[1],:x.shape[0]], x, self.experts_fc2)
        else:
            x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, cross_attn_weights, self_attn_weights

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        # update layer norms
        layer_norm_map = {
            "0": "self_attn_layer_norm",
            "1": "encoder_attn_layer_norm",
            "2": "final_layer_norm",
        }
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
                if "{}.{}.{}".format(name, new, m) not in state_dict and "{}.{}".format(new, m) in self.state_dict():
                    state_dict["{}.{}.{}".format(name, new, m)] = self.state_dict()["{}.{}".format(new, m)]

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                for i in range(self.experts_num):
                    if param_name.find("experts_fc1"+"."+str(i)) != -1:
                        state_dict[prefix + param_name] = self.state_dict()[param_name.replace("experts_fc1"+"."+str(i), "fc1")]
                    if param_name.find("experts_fc2"+"."+str(i)) != -1:
                        state_dict[prefix + param_name] = self.state_dict()[param_name.replace("experts_fc2"+"."+str(i), "fc2")]
                if param_name.find("experts_fc1") == -1 and param_name.find("experts_fc2") == -1:
                    state_dict[prefix + param_name] = self.state_dict()[param_name]
