# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.logging import metrics
from ofasys.module import utils
from ofasys.preprocessor.data_utils import lengths_to_mask

from .base import BaseCriterion, CriterionConfig

logger = logging.getLogger(__name__)


@dataclass
class Tacotron2CriterionConfig(CriterionConfig):
    bce_pos_weight: float = field(
        default=1.0,
        metadata={"help": "weight of positive examples for BCE loss"},
    )
    use_guided_attention_loss: bool = field(
        default=False,
        metadata={"help": "use guided attention loss"},
    )
    guided_attention_loss_sigma: float = field(
        default=0.4,
        metadata={"help": "weight of positive examples for BCE loss"},
    )
    ctc_weight: float = field(default=0.0, metadata={"help": "weight for CTC loss"})
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
    )


class GuidedAttentionLoss(torch.nn.Module):
    """
    Efficiently Trainable Text-to-Speech System Based on Deep Convolutional
    Networks with Guided Attention (https://arxiv.org/abs/1710.08969)
    """

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_weight(s_len, t_len, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(t_len), torch.arange(s_len))
        grid_x = grid_x.to(s_len.device)
        grid_y = grid_y.to(s_len.device)
        w = (grid_y.float() / s_len - grid_x.float() / t_len) ** 2
        return 1.0 - torch.exp(-w / (2 * (sigma**2)))

    def _get_weights(self, src_lens, tgt_lens):
        bsz, max_s_len, max_t_len = len(src_lens), max(src_lens), max(tgt_lens)
        weights = torch.zeros((bsz, max_t_len, max_s_len))
        for i, (s_len, t_len) in enumerate(zip(src_lens, tgt_lens)):
            weights[i, :t_len, :s_len] = self._get_weight(s_len, t_len, self.sigma)
        return weights

    @staticmethod
    def _get_masks(src_lens, tgt_lens):
        in_masks = lengths_to_mask(src_lens)
        out_masks = lengths_to_mask(tgt_lens)
        return out_masks.unsqueeze(2) & in_masks.unsqueeze(1)

    def forward(self, attn, src_lens, tgt_lens, reduction="mean"):
        weights = self._get_weights(src_lens, tgt_lens).to(attn.device)
        masks = self._get_masks(src_lens, tgt_lens).to(attn.device)
        loss = (weights * attn.transpose(1, 2)).masked_select(masks)
        loss = torch.sum(loss) if reduction == "sum" else torch.mean(loss)
        return loss


@register_config("ofasys.criterion", "ofa_tacotron2", Tacotron2CriterionConfig)
class OFATacotron2Criterion(BaseCriterion):
    def __init__(self, task, cfg: Tacotron2CriterionConfig):
        super().__init__(task, cfg)
        self.sentence_avg = cfg.sentence_avg
        self.bce_pos_weight = cfg.bce_pos_weight

        self.guided_attn = None
        if cfg.use_guided_attention_loss:
            self.guided_attn = GuidedAttentionLoss(cfg.guided_attention_loss_sigma)
        self.ctc_weight = cfg.ctc_weight

    def forward(self, model, sample, reduction="mean", update_num=None, pad=1, eos=2):

        bsz, max_len, _ = sample["target"].size()
        feat_tgt = sample["target"]
        feat_len = sample["target_lengths"].view(bsz, 1).expand(-1, max_len)
        eos_tgt = torch.arange(max_len).to(sample["target"].device)
        eos_tgt = eos_tgt.view(1, max_len).expand(bsz, -1)
        eos_tgt = (eos_tgt == (feat_len - 1)).float()

        net_input = sample["net_input"]
        for slot in net_input['slots']:
            if slot.modality == ModalityType.TEXT and slot.is_src:
                src_tokens = slot.value
                src_lens = (src_tokens.ne(eos) & src_tokens.ne(pad)).long().sum(dim=1)
            if slot.modality == ModalityType.AUDIO and slot.is_src:
                src_tokens = slot.value["fbank"]
                src_lens = slot.value["fbank_lengths"]

        tgt_lens = sample["target_lengths"]

        feat_out, extra = model(**sample["net_input"])
        eos_out = extra["eos_out"]

        l1_loss, mse_loss, eos_loss = self.compute_loss(
            extra["feature_out"],
            feat_out,
            eos_out,
            feat_tgt,
            eos_tgt,
            tgt_lens,
            reduction,
        )
        attn_loss = torch.tensor(0.0).type_as(l1_loss)
        if self.guided_attn is not None:
            attn_loss = self.guided_attn(extra["attn"], src_lens, tgt_lens, reduction)
        ctc_loss = torch.tensor(0.0).type_as(l1_loss)
        if self.ctc_weight > 0.0:
            net_output = (feat_out, eos_out, extra)
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.transpose(0, 1)  # T x B x C
            src_mask = lengths_to_mask(src_lens)
            src_tokens_flat = src_tokens.masked_select(src_mask)
            ctc_loss = (
                F.ctc_loss(
                    lprobs,
                    src_tokens_flat,
                    tgt_lens,
                    src_lens,
                    reduction=reduction,
                    zero_infinity=True,
                )
                * self.ctc_weight
            )
        loss = l1_loss + mse_loss + eos_loss + attn_loss + ctc_loss

        sample_size = sample["nsentences"] if self.sentence_avg else sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "l1_loss": utils.item(l1_loss.data),
            "mse_loss": utils.item(mse_loss.data),
            "eos_loss": utils.item(eos_loss.data),
            "attn_loss": utils.item(attn_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
        }
        return loss * self.weight, sample_size, logging_output

    def compute_loss(
        self,
        feat_out,
        feat_out_post,
        eos_out,
        feat_tgt,
        eos_tgt,
        tgt_lens,
        reduction="mean",
    ):
        mask = lengths_to_mask(tgt_lens)
        _eos_out = eos_out[mask].squeeze()
        _eos_tgt = eos_tgt[mask]
        _feat_tgt = feat_tgt[mask]
        _feat_out = feat_out[mask]
        _feat_out_post = feat_out_post[mask]

        l1_loss = F.l1_loss(_feat_out, _feat_tgt, reduction=reduction) + F.l1_loss(
            _feat_out_post, _feat_tgt, reduction=reduction
        )
        mse_loss = F.mse_loss(_feat_out, _feat_tgt, reduction=reduction) + F.mse_loss(
            _feat_out_post, _feat_tgt, reduction=reduction
        )
        eos_loss = F.binary_cross_entropy_with_logits(
            _eos_out,
            _eos_tgt,
            pos_weight=torch.tensor(self.bce_pos_weight),
            reduction=reduction,
        )
        return l1_loss, mse_loss, eos_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]], prefix_name=None) -> None:

        task_name = prefix_name + '/' if prefix_name else ''
        ns = [log.get("sample_size", 0) for log in logging_outputs]
        ntot = sum(ns)
        ws = [n / (ntot + 1e-8) for n in ns]

        for key in ["loss", "l1_loss", "mse_loss", "eos_loss", "attn_loss", "ctc_loss"]:
            vals = [log.get(key, 0) for log in logging_outputs]
            val = sum(val * w for val, w in zip(vals, ws))
            metrics.log_scalar(f"{task_name}{key}", val, ntot, round=3)
            if key == "loss":
                metrics.log_scalar("loss", val, ntot, priority=0, round=3)
        metrics.log_scalar(f"{task_name}sample_size", ntot, len(logging_outputs))

        # inference metrics
        if "targ_frames" not in logging_outputs[0]:
            return
        n = sum(log.get("targ_frames", 0) for log in logging_outputs)
        for key, new_key in [
            ("mcd_loss", "mcd_loss"),
            ("pred_frames", "pred_ratio"),
            ("nins", "ins_rate"),
            ("ndel", "del_rate"),
        ]:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(f"{task_name}{new_key}", val / n, n, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
