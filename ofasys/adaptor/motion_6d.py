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
from ofasys.module.diffusion import ContinuousTimeMLP
from ofasys.module.motion_6d import discrete_fourier_transform_v3
from ofasys.preprocessor import Dictionary

logger = logging.getLogger(__name__)


@dataclass
class Motion6dAdaptorConfig(TextAdaptorConfig):
    float_embed_dim: int = field(default=256, metadata={"help": ""})
    frame_embed_dim: int = field(default=256, metadata={"help": ""})


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

        data_dim = 3 + 39 * 6  # TODO(jianxin): Adaptor need access to Preprocessor to get this kind of information.
        self.frame_encoder = nn.Linear(data_dim, cfg.frame_embed_dim)
        self.frame_to_embed = nn.Linear(cfg.frame_embed_dim, cfg.embed_dim)
        self.embed_to_frame = nn.Linear(cfg.embed_dim, cfg.frame_embed_dim)
        self.frame_decoder = nn.Linear(cfg.frame_embed_dim, data_dim)

        self.noise_level_emb = ContinuousTimeMLP(cfg.embed_dim * 2, learned_sinusoidal_dim=cfg.float_embed_dim)

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
        dtype, device = input_value.dtype, input_value.device

        padding_masks = slot.value['masks']
        assert padding_masks.shape == (batch_size, seq_len)

        token_embed = self.frame_to_embed(self.frame_encoder(input_value))
        embed_dim = token_embed.shape[-1]

        if 'noise_level' in slot.value:
            noise_level = slot.value['noise_level'].to(dtype=dtype)
            noise_embed = self.noise_level_emb(noise_level).unsqueeze(1)  # [B,1,D*2]
            noise_scale, noise_shift = noise_embed.chunk(chunks=2, dim=-1)
            token_embed = noise_scale * token_embed + noise_shift
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
        out = self.embed_to_frame(x)
        assert 'latent' not in slot.value
        slot.value['latent'] = out
        out = self.frame_decoder(out)
        assert out.shape == (batch_size, seq_len, data_dim)
        return out, extra

    def _custom_reg_loss(self, slot: Slot, prediction, target):  # rename it to custom_reg_loss to enable it
        # Benefits of frequency-domain regularization:
        #  (1) Prevent jitter by filtering out the unusual high frequency noises.
        #  (2) Easier to discover periodic patterns.
        #  (3) Related with the NPSS metric.

        batch_size, seq_len, embed_dim = prediction.shape
        if not (batch_size >= 2):
            logger.warning('Contrastive learning requires batch_size >= 2.')
            return torch.zeros([], dtype=prediction.dtype, device=prediction.device)

        padding_masks = slot.value['masks']
        assert padding_masks.shape == (batch_size, seq_len)
        num_pads, _ = padding_masks.type(dtype=torch.int).sum(-1).max(dim=0)  # [B,T]->[B]->scalar
        seq_len = seq_len - num_pads.item()

        if not (seq_len >= 4):
            # logger.warning('Frequency regularization is valid only if seq_len >= 4.')
            return torch.zeros([], dtype=prediction.dtype, device=prediction.device)

        # Picking the first few time steps can be interpreted as DFT with Windowing.
        pred_freq = discrete_fourier_transform_v3(slot.value['latent'][:, :seq_len], get_unnorm_sqr_amp=True)
        with torch.no_grad():  # SimSiam, see https://arxiv.org/pdf/2011.10566.pdf
            true_freq = discrete_fourier_transform_v3(self.frame_encoder(target[:, :seq_len]), get_unnorm_sqr_amp=True)
            true_freq = true_freq.detach()

        # The contrastive loss' benefits:
        #   (1) Prevent the encoder from collapsing, i.e., outputting the same value regardless of the inputs.
        #   (2) Normalization prevents it from neglecting the frequency components of small magnitudes.
        #     With real world data, a few frequency components can have way larger magnitudes than the rest.
        axis = 1  # axis=2
        eps = 1e-6
        pred_freq, true_freq = pred_freq + eps, true_freq + eps  # torch.sqrt has NaN grad at 0
        pred_freq = torch.sqrt(pred_freq / pred_freq.sum(dim=axis, keepdim=True))  # l2-normalized along axis
        true_freq = torch.sqrt(true_freq / true_freq.sum(dim=axis, keepdim=True))  # l2-normalized along axis
        logits = torch.sum(pred_freq * true_freq, dim=axis)  # [B,D]
        loss = 1.0 - logits.mean()
        return loss
