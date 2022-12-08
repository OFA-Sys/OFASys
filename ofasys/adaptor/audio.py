# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ofasys import ModalityType
from ofasys.adaptor.base import AdaptorOutput, BaseAdaptor, BaseAdaptorConfig, Slot
from ofasys.configure import ChoiceEnum, register_config
from ofasys.distributed import fsdp_wrap
from ofasys.module import (
    Conv2dSubsampling4,
    Embedding,
    EncDecBaseConfig,
    LayerDropModuleList,
    checkpoint_wrapper,
    utils,
)
from ofasys.preprocessor import Dictionary

DEFAULT_MAX_WAV_POSITIONS = 4096
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class DownConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.act = nn.GLU()
        self.conv2 = nn.Conv1d((out_channels // 2), out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x.permute(0, 2, 1))
        x = nn.functional.glu(x, dim=1)
        x = self.conv2(x).permute(0, 2, 1)
        return x


def make_audio_bucket_position(bucket_size, max_position=DEFAULT_MAX_WAV_POSITIONS):
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
class AudioFbankAdaptorConfig(BaseAdaptorConfig):
    output_frame_dim: int = field(default=80, metadata={"help": "output_frame_dim"})
    n_frames_per_step: int = field(default=1, metadata={"help": "n_frames_per_step"})

    is_transformer_layers: bool = field(
        default=False, metadata={"help": "whether encoder prenet have transformer net"}
    )
    encoder_config: EncDecBaseConfig = EncDecBaseConfig()
    encode_drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "encoder drop path rate"},
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )
    attn_scale_factor: float = field(
        default=2,
        metadata={"help": "attention scale factor"},
    )
    scale_attn: bool = field(
        default=True,
        metadata={"help": "scale attn"},
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
    use_fused: bool = field(
        default=True,
        metadata={"help": "use fused op"},
    )

    # decoder prenet
    prenet_layers: int = field(default=2, metadata={"help": "prenet layers"})
    prenet_dim: int = field(default=256, metadata={"help": "prenet dim"})
    prenet_dropout: float = field(default=0.5, metadata={"help": "prenet dropout"})
    # decoder postnet
    postnet_conv_dim: int = field(default=512, metadata={"help": "postnet_conv_dim"})
    postnet_conv_kernel_size: int = field(default=5, metadata={"help": "postnet_conv_kernel_size"})
    postnet_layers: int = field(default=5, metadata={"help": "postnet_layers"})
    postnet_dropout: float = field(default=0.5, metadata={"help": "postnet_dropout"})

    # masking
    use_mask: bool = field(default=False, metadata={"help": "use mask"})
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(default=False, metadata={"help": "whether to allow masks to overlap"})
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False

    require_same_masks: bool = field(
        default=True,
        metadata={"help": "whether to number of masked timesteps must be the same across all " "examples in a batch"},
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )


@register_config("ofasys.adaptor", "audio_fbank", AudioFbankAdaptorConfig)
class AudioFbankAdaptor(BaseAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: AudioFbankAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)

        self.audio_bucket_size = cfg.max_position

        self.out_dim = cfg.output_frame_dim * cfg.n_frames_per_step
        # fbank encoder prenet
        self.subsample = Conv2dSubsampling4(self.out_dim, cfg.embed_dim)
        # encoder transformer
        self.is_transformer_layers = cfg.is_transformer_layers
        if cfg.is_transformer_layers:
            if cfg.encoder_config.layerdrop > 0.0:
                self.transformer_layers = LayerDropModuleList(p=cfg.encoder_config.layerdrop)
            else:
                self.transformer_layers = nn.ModuleList([])

            dpr = torch.linspace(0, cfg.encode_drop_path_rate, cfg.encoder_config.layers)
            self.transformer_layers.extend(
                [self.build_encoder_layer(cfg, drop_path_rate=dpr[i]) for i in range(cfg.encoder_config.layers)]
            )
            self.register_forward_hook(AudioFbankAdaptor.forward_hook_fn)

        # fbank decoder prenet
        self.prenet = nn.Sequential(
            Prenet(self.out_dim, cfg.prenet_layers, cfg.prenet_dim, cfg.prenet_dropout),
            nn.Linear(cfg.prenet_dim, cfg.embed_dim),
        )

        self.embed_audio_positions = Embedding(cfg.max_position, cfg.embed_dim)

        audio_num_rel_dis = 2 * self.audio_bucket_size - 1
        audio_rp_bucket = make_audio_bucket_position(self.audio_bucket_size)

        num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
        self.audio_rel_pos_table_list = nn.ModuleList(
            [Embedding(audio_num_rel_dis, cfg.num_attention_heads, zero_init=True) for _ in range(num_rel_pos_tables)]
        )
        self.register_buffer("audio_rp_bucket", audio_rp_bucket)

        self.n_frames_per_step = cfg.n_frames_per_step
        self.out_dim = cfg.output_frame_dim * cfg.n_frames_per_step
        self.feat_proj = nn.Linear(cfg.embed_dim, self.out_dim)
        self.eos_proj = nn.Linear(cfg.embed_dim, 1)

        # fbank decoder postnet
        self.postnet = Postnet(
            self.out_dim,
            cfg.postnet_conv_dim,
            cfg.postnet_conv_kernel_size,
            cfg.postnet_layers,
            cfg.postnet_dropout,
        )

        # encoder use mask
        self.use_mask = cfg.use_mask
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.embed_dim).uniform_())
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.mask_channel_before = cfg.mask_channel_before

        self.require_same_masks = cfg.require_same_masks
        self.mask_dropout = cfg.mask_dropout

    def get_rel_pos_bias(self, batch_size, seq_length, idx, **kwargs):
        rp_bucket = self.audio_rp_bucket[:seq_length, :seq_length]
        values = F.embedding(rp_bucket, self.audio_rel_pos_table_list[idx].weight)
        return values

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot): ModalityType.AUDIO
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
        assert slot.modality == ModalityType.AUDIO

        if slot.is_src:
            fbank = slot.value["fbank"]
            fbank_lengths = slot.value["fbank_lengths"]
            mask_indices = slot.value.get('mask_indices', None)
            mask_channel_indices = slot.value.get('mask_channel_indices', None)

            fbank_feature, fbank_feature_length = self.subsample(fbank, fbank_lengths)
            fbank_feature_padding_mask = (
                torch.zeros(fbank_feature.shape[:2], device=fbank_feature.device).bool()
                # torch.BoolTensor(fbank_feature.shape[:2], device=fbank_feature.device).fill_(False)
                # if self.pad_audio else None
            )
            for i, l in enumerate(fbank_feature_length):
                diff = l - fbank_feature_padding_mask.shape[-1]
                if diff < 0:
                    fbank_feature_padding_mask[i, diff:] = True

            fbank_pos_embed = self.embed_audio_positions(utils.new_arange(fbank_feature, *fbank_feature.size()[:2]))

            if (slot.has_attr("use_mask") or self.use_mask) and mask_indices is not None:
                masked_fbank_feature = self.apply_mask(
                    fbank_feature,
                    fbank_feature_padding_mask,
                    mask_indices=mask_indices,
                    mask_channel_indices=mask_channel_indices,
                )

                return AdaptorOutput(masked_fbank_feature, fbank_feature_padding_mask, fbank_pos_embed, [])

            else:
                return AdaptorOutput(fbank_feature, fbank_feature_padding_mask, fbank_pos_embed, [])

        else:
            fbank = slot.value["fbank"]
            fbank_lengths = slot.value["fbank_lengths"]

            fbank_feature = self.prenet(fbank)
            fbank_feature_padding_mask = lengths_to_padding_mask(fbank_lengths)

            fbank_pos_embed = self.embed_audio_positions(utils.new_arange(fbank_feature, *fbank_feature.size()[:2]))

            return AdaptorOutput(fbank_feature, fbank_feature_padding_mask, fbank_pos_embed, [])

    def build_encoder_layer(self, cfg, drop_path_rate=0.0):
        encoder_dict = dict()
        for k in cfg.encoder_config.__dataclass_fields__.keys():
            encoder_dict[k] = getattr(cfg.encoder_config, k)
        cfg.encoder = encoder_dict
        from ofasys.model.transformer_layer import TransformerEncoderLayer

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

    def forward_hook_fn(self, inputs, output: AdaptorOutput):
        slot: Slot = inputs[0]
        embed = self.embed_scale * output.embed
        if self.cfg.entangle_position_embedding and output.pos_embed is not None:
            embed += output.pos_embed
        if slot.is_src and self.type_embedding is not None:
            embed += self.type_embedding.weight.squeeze()
        if self.layernorm_embedding is not None:
            embed = self.layernorm_embedding(embed)
        if self.layernorm_position is not None and output.pos_embed is not None:
            output.pos_embed = self.layernorm_position(output.pos_embed)
        output.embed = self.dropout_module(embed)

        if not output.self_attn_bias:
            output.self_attn_bias = []
            batch_size, seq_length = output.embed.size()[:2]
            for idx in range(self.num_layers):
                values = self.get_rel_pos_bias(batch_size, seq_length, idx)
                output.self_attn_bias.append(self.expand_rel_pos_bias(values, batch_size, slot.is_src))

        if self.is_transformer_layers:
            # B x T x C -> T x B x C
            x = output.embed.transpose(0, 1)
            has_pad = output.masks.any()
            if has_pad:
                output.embed *= 1 - output.masks.unsqueeze(-1).type_as(output.embed)

            self_attn_bias_array = []
            batch_size, seq_length = output.embed.size()[:2]
            for idx in range(self.cfg.encoder_config.layers):
                values = self.get_rel_pos_bias(batch_size, seq_length, idx)
                self_attn_bias_array.append(self.expand_rel_pos_bias(values, batch_size, slot.is_src))

            # encoder layers
            for idx, layer in enumerate(self.transformer_layers):
                self_attn_bias = self_attn_bias_array[idx].view(-1, x.size(0), x.size(0))
                x, _ = layer(
                    x,
                    encoder_padding_mask=output.masks if has_pad else None,
                    self_attn_bias=self_attn_bias,
                )
            output.embed = x.transpose(0, 1)

        return output

    def get_mask_indices(self, B, T, C, mask_prob=None):
        mask_indices = None
        mask_channel_indices = None
        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, T, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )

        if self.mask_prob > 0 or mask_prob is not None:
            if mask_prob is None:
                mask_prob = self.mask_prob
            mask_indices = compute_mask_indices(
                (B, T),
                # TODO: padding mask is need here. temp remove it
                None,
                mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=1,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
                require_same_masks=self.require_same_masks,
                mask_dropout=self.mask_dropout,
            )
        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
        if mask_indices is not None:
            mask_indices = torch.from_numpy(mask_indices)

        if mask_channel_indices is not None:
            mask_channel_indices = torch.from_numpy(mask_channel_indices)
        return mask_indices, mask_channel_indices

    def apply_mask(self, x, padding_mask, mask_indices=None, mask_channel_indices=None, mask_prob=None):
        B, T, C = x.shape
        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = mask_channel_indices.to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        if self.mask_prob > 0 or mask_prob is not None:
            mask_indices = mask_indices.to(x.device)
            x = utils.index_put(x, mask_indices, self.mask_emb)

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            mask_channel_indices = mask_channel_indices.to(x.device).unsqueeze(1).expand(-1, T, -1)
            x = utils.index_put(x, mask_channel_indices, 0)

        return x

    def forward_output(self, x: Tensor, extra: Dict[str, Any], slot: Slot, **kwargs):
        attn = [extra["attn"][0].transpose(2, 1)]
        feat_out = self.feat_proj(x)
        bsz, seq_len, _ = x.size()
        eos_out = self.eos_proj(x)
        post_feat_out = feat_out + self.postnet(feat_out)
        extra["attn"] = attn
        extra["eos_out"] = eos_out
        extra["feature_out"] = feat_out
        return post_feat_out, extra


@dataclass
class AudioTargetFbankAdaptorConfig(BaseAdaptorConfig):
    output_frame_dim: int = field(default=80, metadata={"help": "output_frame_dim"})
    n_frames_per_step: int = field(default=1, metadata={"help": "n_frames_per_step"})
    conv_kernel_size: int = field(default=5, metadata={"help": "conv_kernel_size"})
    prenet_layers: int = field(default=2, metadata={"help": "prenet layers"})
    prenet_dim: int = field(default=256, metadata={"help": "prenet dim"})
    prenet_dropout: float = field(default=0.5, metadata={"help": "prenet dropout"})
    postnet_conv_dim: int = field(default=512, metadata={"help": "postnet_conv_dim"})
    postnet_conv_kernel_size: int = field(default=5, metadata={"help": "postnet_conv_kernel_size"})
    postnet_layers: int = field(default=5, metadata={"help": "postnet_layers"})
    postnet_dropout: float = field(default=0.5, metadata={"help": "postnet_dropout"})


@register_config("ofasys.adaptor", "audio_tgt_fbank", AudioTargetFbankAdaptorConfig)
class AudioTargetFbankAdaptor(BaseAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: AudioTargetFbankAdaptorConfig,
    ):

        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)

        self.audio_bucket_size = cfg.max_position
        # fbank encoder
        self.out_dim = cfg.output_frame_dim * cfg.n_frames_per_step
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.prenet = nn.Sequential(
            Prenet(self.out_dim, cfg.prenet_layers, cfg.prenet_dim, cfg.prenet_dropout),
            nn.Linear(cfg.prenet_dim, cfg.embed_dim),
        )

        self.embed_audio_positions = Embedding(cfg.max_position, cfg.embed_dim)

        audio_num_rel_dis = 2 * self.audio_bucket_size - 1
        audio_rp_bucket = make_audio_bucket_position(self.audio_bucket_size)
        self.audio_rel_pos_table_list = nn.ModuleList(
            [Embedding(audio_num_rel_dis, cfg.num_attention_heads, zero_init=True) for _ in range(self.num_layers)]
        )
        self.register_buffer("audio_rp_bucket", audio_rp_bucket)

        self.n_frames_per_step = cfg.n_frames_per_step
        self.out_dim = cfg.output_frame_dim * cfg.n_frames_per_step
        self.feat_proj = nn.Linear(cfg.embed_dim, self.out_dim)
        self.eos_proj = nn.Linear(cfg.embed_dim, 1)

        self.postnet = Postnet(
            self.out_dim,
            cfg.postnet_conv_dim,
            cfg.postnet_conv_kernel_size,
            cfg.postnet_layers,
            cfg.postnet_dropout,
        )

    def get_rel_pos_bias(self, batch_size, seq_length, idx, **kwargs):
        rp_bucket = self.audio_rp_bucket[:seq_length, :seq_length]
        values = F.embedding(rp_bucket, self.audio_rel_pos_table_list[idx].weight)
        return values

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot): ModalityType.AUDIO
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
        assert slot.modality == ModalityType.AUDIO

        fbank = slot.value["fbank"]
        fbank_lengths = slot.value["fbank_lengths"]
        fbank_feature = self.prenet(fbank)
        fbank_feature_padding_mask = lengths_to_padding_mask(fbank_lengths)

        fbank_pos_embed = self.embed_audio_positions(utils.new_arange(fbank_feature, *fbank_feature.size()[:2]))

        return AdaptorOutput(fbank_feature, fbank_feature_padding_mask, fbank_pos_embed, [])

    def forward_output(self, x: Tensor, extra: Dict[str, Any], slot: Slot, **kwargs):

        attn = [extra["attn"][0].transpose(2, 1)]
        feat_out = self.feat_proj(x)
        bsz, seq_len, _ = x.size()
        eos_out = self.eos_proj(x)
        post_feat_out = feat_out + self.postnet(feat_out)
        extra["attn"] = attn
        extra["eos_out"] = eos_out
        extra["feature_out"] = feat_out
        return post_feat_out, extra


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(mask_idc, len(mask_idc) - num_holes, replace=False)

        mask[i, mask_idc] = True

    return mask


# lens: torch.LongTensor
# returns: torch.BoolTensor
def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class Prenet(nn.Module):
    def __init__(self, in_dim, n_layers, n_units, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(in_dim if i == 0 else n_units, n_units), nn.ReLU()) for i in range(n_layers)
        )
        self.dropout = dropout

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout)  # always applies dropout
        return x


class Postnet(nn.Module):
    def __init__(self, in_dim, n_channels, kernel_size, n_layers, dropout):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        assert kernel_size % 2 == 1
        for i in range(n_layers):
            cur_layers = (
                [
                    nn.Conv1d(
                        in_dim if i == 0 else n_channels,
                        n_channels if i < n_layers - 1 else in_dim,
                        kernel_size=kernel_size,
                        padding=((kernel_size - 1) // 2),
                    ),
                    nn.BatchNorm1d(n_channels if i < n_layers - 1 else in_dim),
                ]
                + ([nn.Tanh()] if i < n_layers - 1 else [])
                + [nn.Dropout(dropout)]
            )
            nn.init.xavier_uniform_(
                cur_layers[0].weight, torch.nn.init.calculate_gain("tanh" if i < n_layers - 1 else "linear")
            )
            self.convolutions.append(nn.Sequential(*cur_layers))

    def forward(self, x):
        x = x.transpose(1, 2)  # B x T x C -> B x C x T
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)
