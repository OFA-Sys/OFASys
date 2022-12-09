# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ofasys.adaptor.general import OFAAdaptorConfig
from ofasys.configure import register_config
from ofasys.distributed import fsdp_wrap
from ofasys.module import TransformerConfig, init_bert_params, utils
from ofasys.preprocessor import Dictionary, Slot

from .fairseq_model import FairseqEncoderDecoderModel
from .transformer import TransformerDecoder, TransformerEncoder

logger = logging.getLogger(__name__)


@dataclass
class GeneralistModelConfig(TransformerConfig):
    arch: str = field(
        default='base',
        metadata={"help": "model arch"},
    )
    encode_drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "encoder drop path rate"},
    )
    decode_drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "decoder drop path rate"},
    )

    attn_scale_factor: float = field(
        default=2,
        metadata={"help": "attention scale factor"},
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "freeze encoder"},
    )
    freeze_encoder_embedding: bool = field(
        default=False,
        metadata={"help": "freeze encoder token embedding"},
    )
    freeze_decoder_embedding: bool = field(
        default=False,
        metadata={"help": "freeze decoder token embedding"},
    )
    add_type_embedding: bool = field(
        default=True,
        metadata={"help": "add source/region/patch type embedding"},
    )
    entangle_position_embedding: bool = field(
        default=False,
        metadata={"help": "entangle position embedding"},
    )
    sync_bn: bool = field(
        default=False,
        metadata={"help": "sync batchnorm"},
    )

    scale_attn: bool = field(
        default=True,
        metadata={"help": "scale attention"},
    )
    scale_fc: bool = field(
        default=True,
        metadata={"help": "scale fc"},
    )
    scale_heads: bool = field(
        default=True,
        metadata={"help": "scale heads"},
    )
    scale_resids: bool = field(
        default=False,
        metadata={"help": "scale resids"},
    )

    checkpoint_adaptor_activations: bool = field(
        default=False,
        metadata={"help": "apply checkpointing activation for adaptors"},
    )
    use_fused: bool = field(
        default=False,
        metadata={"help": "use fused kernel"},
    )
    use_self_attn_bias: bool = field(
        default=True,
        metadata={"help": "use self-attn-bias"},
    )
    adaptor: OFAAdaptorConfig = OFAAdaptorConfig()
    share_attn_bias: bool = field(
        default=False,
        metadata={"help": "whether to share attn_bias cross transformer layers"},
    )


@register_config("ofasys.model", "unify", dataclass=GeneralistModelConfig)
class GeneralistModel(FairseqEncoderDecoderModel):
    __jit_unused_properties__ = ["supported_targets"]

    def __init__(self, cfg: GeneralistModelConfig = None):
        if cfg is None:
            cfg = GeneralistModelConfig.from_yaml(
                os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    'config',
                    'default_model.yaml',
                )
            )
        self.cfg = cfg

        if cfg.encoder_layers_to_keep:
            cfg.encoder_layers = len(cfg.encoder_layers_to_keep.split(","))
        if cfg.decoder_layers_to_keep:
            cfg.decoder_layers = len(cfg.decoder_layers_to_keep.split(","))
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        if cfg.arch:
            arch_func = eval('ofa_arch_' + cfg.arch)
            arch_func(cfg)

    @classmethod
    def from_yaml(cls, yaml_path):
        return GeneralistModel(GeneralistModelConfig.from_yaml(yaml_path))

    def initialize(self, global_dict: Dictionary):
        encoder = TransformerEncoder(self.cfg, global_dict)
        decoder = TransformerDecoder(self.cfg, global_dict, self.cfg.no_cross_attention)
        if not self.cfg.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=self.cfg.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=self.cfg.min_params_to_wrap)

        super().__init__(encoder, decoder)
        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        if self.cfg.freeze_encoder:
            self.encoder.requires_grad_(False)

        self.global_dict = global_dict

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        slots: List[Slot],
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
        return_encoder_out: bool = False,
        return_hf_dict: bool = False,
        return_all_attention_weights: bool = False,
    ):
        """
        Args:
            slots (List[Slot]): preprocessed data.
            features_only (bool, optional): only return features without
                applying ``adaptor.forward_output()`` (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            return_encoder_out (bool, optional): also return encoder output (default: False).
            return_hf_dict (bool, optional): return a dict like huggingface style instead of a tuple (default False).
            return_all_attention_weights (bool, optional): also return all attention weights (default: False).

        Returns:
            if **return_hf_dict** is True, return a hf-style dict else a tuple:
            tuple:

                - the decoder's output: the decoder's features of shape ``(batch, tgt_len, embed_dim)``
                  if *features_only* is True, else return outputs from adaptor.
                - a dictionary with decoder extra outputs.

                    - **attn** (List[Tensor]) : return specific attention weights.
                    - **inner_states** (List[Tensor]): all intermediate encoder hidden states
                      of shape ``(tgt_len, batch, embed_dim)``.
                    - **decoder_attentions** (List[Tensor]): attention weights of decoder's self attention of
                      shape ``(num_heads, batch_size, tgt_len, tgt_len)``.
                      Only return if *return_all_attention_weights* is True.
                    - **cross_attentions** (List[Tensor]): attention weights of decoder's self attention of
                      shape ``(num_heads, batch_size, src_len, tgt_len)``.
                      Only return if *return_all_attention_weights* is True.
                - a dictionary with encoder outputs (if return_encoder_out).

            dict:
                - **last_hidden_state** (Tensor): the last decoder layer's output of
                  shape ``(batch, tgt_len, embed_dim)``
                - **decoder_adaptor_out** (Tensor): the last decoder layer's output after applying
                  ``adaptor.forward_output()``.
                - **decoder_attentions** (List[Tensor]): attention weights of decoder's self attention of
                  shape ``(num_heads, batch_size, tgt_len, tgt_len)``.
                  Only return if *return_all_attention_weights* is True.
                - **cross_attentions** (List[Tensor]): attention weights of decoder's self attention of
                  shape ``(num_heads, batch_size, src_len, tgt_len)``.
                  Only return if *return_all_attention_weights* is True.
                - **decoder_hidden_states** (List[Tensor]): all intermediate
                  decoder hidden states of shape ``(batch, tgt_len, embed_dim)``.
                  Only return if *return_all_hiddens* is True.
                - **encoder_last_hidden_state** (Tensor): the last encoder layer's output of
                  shape ``(src_len, batch, embed_dim)``. Only return if *return_encoder_out* is True.
                - **encoder_attentions** (Tensor): attention weights of encoder's self attention of
                  shape ``(num_heads, batch_size, src_len, src_len)``.
                  Only return if *return_all_attention_weights* and *return_encoder_out* are both True.
                - **encoder_hidden_states** (List[Tensor]): all intermediate
                  encoder  hidden states of shape ``(batch, src_len, embed_dim)``,
                  Only return if *return_all_hiddens* and *return_encoder_out* are both True.

        """
        encoder_out = self.encoder(
            list(filter(lambda slot: slot.is_src, slots)),
            return_all_hiddens=return_all_hiddens,
            return_all_attention_weights=return_all_attention_weights,
        )

        decoder_out, decoder_extra_out = self.decoder(
            list(filter(lambda slot: not slot.is_src, slots)),
            encoder_out=encoder_out,
            features_only=features_only,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            return_all_hiddens=return_all_hiddens,
            return_all_attention_weights=return_all_attention_weights,
        )
        if return_hf_dict:
            ret = {
                "last_hidden_state": decoder_extra_out['last_hidden_state'],
            }
            if return_all_attention_weights:
                ret["decoder_attentions"] = decoder_extra_out['decoder_attentions']
                ret["cross_attentions"] = decoder_extra_out['cross_attentions']

            if return_all_hiddens:
                ret["decoder_hidden_states"] = decoder_extra_out['inner_states']

            if not features_only:
                ret["decoder_adaptor_out"] = decoder_out

            if return_encoder_out:
                ret["encoder_last_hidden_state"] = encoder_out["encoder_out"]
                if return_all_attention_weights:
                    ret["encoder_attentions"] = encoder_out["encoder_attention_weights"]
                if return_all_hiddens:
                    ret["encoder_hidden_states"] = encoder_out["encoder_states"]
            return ret

        else:
            if return_encoder_out:
                return decoder_out, decoder_extra_out, encoder_out
            else:
                return decoder_out, decoder_extra_out

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = self.get_logits_from_net_output(net_output)
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_logits_from_net_output(self, net_output):
        if isinstance(net_output, Dict):
            return net_output['decoder_adaptor_out']
        else:
            return net_output[0]

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        # remove outdated params for backward compatibility when loading old checkpoints
        del_keys = ["decoder.output_projection.weight"]
        if not self.cfg.use_self_attn_bias:
            del_keys += [
                'decoder.cross_pos_q_linear.weight',
                'decoder.cross_pos_q_linear.bias',
                'encoder.adaptor.pos_q_linear.weight',
                'encoder.adaptor.pos_q_linear.bias',
                'decoder.adaptor.pos_q_linear.weight',
                'decoder.adaptor.pos_q_linear.bias',
                'decoder.cross_pos_k_linear.weight',
                'decoder.cross_pos_k_linear.bias',
                'encoder.adaptor.pos_k_linear.weight',
                'encoder.adaptor.pos_k_linear.bias',
                'decoder.adaptor.pos_k_linear.weight',
                'decoder.adaptor.pos_k_linear.bias',
            ]

        for k in del_keys:
            if k in state_dict:
                logger.info(f'remove {k} from old ckpt.')
                del state_dict[k]

        # extend embed_tokens if the loaded dict is smaller than the current model.
        loaded_dict_size = state_dict["encoder.adaptor.embed_tokens.weight"].size(0)
        if loaded_dict_size < len(self.encoder.dictionary):
            num_tokens_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.adaptor.embed_tokens.weight"].size(1)
            new_token_embed_to_add = torch.zeros(num_tokens_to_add, embed_dim)
            torch.nn.init.normal_(new_token_embed_to_add, mean=0, std=embed_dim**-0.5)
            new_token_embed_to_add = new_token_embed_to_add.to(
                dtype=state_dict["encoder.adaptor.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.adaptor.embed_tokens.weight"] = torch.cat(
                [state_dict["encoder.adaptor.embed_tokens.weight"], new_token_embed_to_add]
            )
            state_dict["decoder.adaptor.embed_tokens.weight"] = torch.cat(
                [state_dict["decoder.adaptor.embed_tokens.weight"], new_token_embed_to_add]
            )

    def update_embedding(self, state):
        assert "global_dict_indices" in state, 'Cannot find global_dict in restored ckpt!'
        loaded_global_dict = state["global_dict_indices"]
        tokens_sorted = sorted(loaded_global_dict.items(), key=lambda x: x[1])
        emb_dim = state["model"]["encoder.adaptor.embed_tokens.weight"].size(1)
        len_dict = len(self.global_dict)
        idx = [self.global_dict.indices.get(token[0], len_dict) for token in tokens_sorted]
        encoder_embedding = torch.zeros(len_dict + 1, emb_dim)
        torch.nn.init.normal_(encoder_embedding, mean=0, std=emb_dim**-0.5)
        encoder_embedding.to(dtype=state["model"]["encoder.adaptor.embed_tokens.weight"].dtype)
        encoder_embedding.index_copy_(0, torch.tensor(idx), state["model"]["encoder.adaptor.embed_tokens.weight"])
        state["model"]["encoder.adaptor.embed_tokens.weight"] = encoder_embedding[:-1, :]
        state["model"]["decoder.adaptor.embed_tokens.weight"] = encoder_embedding[:-1, :]

    def update_sample(self, sample):
        # use for some of the data need to be processed on GPU.
        sample = self.encoder.adaptor.update_sample(sample)
        sample = self.decoder.adaptor.update_sample(sample)
        return sample


def ofa_arch_base(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 768
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 768
    cfg.decoder.input_dim = cfg.decoder.output_dim = 768
    cfg.encoder.layers = cfg.decoder.layers = 6
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 12
    cfg.adaptor.image_resnet.resnet_type = "resnet101"


def ofa_arch_asr_small(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 256
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 2048
    cfg.decoder.input_dim = cfg.decoder.output_dim = 256
    cfg.encoder.layers = 12
    cfg.decoder.layers = 6
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 4
    cfg.adaptor.image_resnet.resnet_type = "resnet101"


def ofa_arch_asr_base(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 768
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 768
    cfg.decoder.input_dim = cfg.decoder.output_dim = 768
    cfg.encoder.layers = 12
    cfg.decoder.layers = 6
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 12
    cfg.adaptor.image_resnet.resnet_type = "resnet101"


def ofa_arch_tiny(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 256
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 256
    cfg.decoder.input_dim = cfg.decoder.output_dim = 256
    cfg.encoder.layers = cfg.decoder.layers = 4
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 4
    cfg.adaptor.image_resnet.resnet_type = "resnet50"


def ofa_arch_medium(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 512
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 512
    cfg.decoder.input_dim = cfg.decoder.output_dim = 512
    cfg.encoder.layers = cfg.decoder.layers = 4
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 8
    cfg.adaptor.image_resnet.resnet_type = "resnet101"


def ofa_arch_large(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 1024
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 1024
    cfg.decoder.input_dim = cfg.decoder.output_dim = 1024
    cfg.encoder.layers = cfg.decoder.layers = 12
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 16
    cfg.adaptor.image_resnet.resnet_type = "resnet152"


def ofa_arch_huge(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 1280
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 1280
    cfg.decoder.input_dim = cfg.decoder.output_dim = 1280
    cfg.encoder.layers = 24
    cfg.decoder.layers = 12
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 16
    cfg.adaptor.image_resnet.resnet_type = "resnet152"


def ofa_arch_6b(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 2560
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 2560
    cfg.decoder.input_dim = cfg.decoder.output_dim = 2560
    cfg.encoder.layers = 36
    cfg.decoder.layers = 24
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 32
    cfg.adaptor.image_resnet.resnet_type = None


def ofa_arch_8b(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 2560
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 2560
    cfg.decoder.input_dim = cfg.decoder.output_dim = 2560
    cfg.encoder.layers = 48
    cfg.decoder.layers = 36
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 32
    cfg.adaptor.image_resnet.resnet_type = None


def ofa_arch_10b(cfg: GeneralistModelConfig):
    cfg.encoder.embed_dim = cfg.decoder.embed_dim = 2816
    cfg.encoder.ffn_embed_dim = cfg.decoder.ffn_embed_dim = 4 * 2816
    cfg.decoder.input_dim = cfg.decoder.output_dim = 2816
    cfg.encoder.layers = 48
    cfg.decoder.layers = 36
    cfg.encoder.attention_heads = cfg.decoder.attention_heads = 32
    cfg.adaptor.image_resnet.resnet_type = None
