# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.engine.criterion.base import BaseCriterion
from ofasys.engine.criterion.tacotron2_loss import (
    OFATacotron2Criterion,
    Tacotron2CriterionConfig,
)
from ofasys.logging import metrics
from ofasys.module import utils


def compute_conv_output_length(hin, kernel_size=3, stride=2):
    return (hin - (kernel_size - 1) - 1) // stride + 1


def build_mask_sample(sample, model):

    slot_mask_indices = []
    for slot in sample["net_input"]["slots"]:
        if slot.is_src:
            if slot.modality == ModalityType.AUDIO:
                adaptor = model.encoder.adaptor.get_adaptor(slot)
                fbank = slot.value['fbank']
                B, T, C = fbank.shape
                sub_sample_T = compute_conv_output_length(compute_conv_output_length(T))
                mask_indices, mask_channel_indices = adaptor.get_mask_indices(B, sub_sample_T, C)
                slot.value['mask_indices'] = mask_indices
                slot.value['mask_channel_indices'] = mask_channel_indices
            else:
                mask_indices = slot.masks.new_zeros(slot.masks.size()).bool()
            slot_mask_indices.append(mask_indices)
    sample['mask_indices'] = torch.cat(slot_mask_indices, dim=1)
    return sample


@dataclass
class SpeechPretrainCriterionConfig(Tacotron2CriterionConfig):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default_factory=lambda: [
            10,
        ],
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )
    mam_weight: float = field(
        default=1.0,
        metadata={"help": "weight of masked audio models (MAM) loss"},
    )
    dec_weight: float = field(
        default=1.0,
        metadata={"help": "weight of tacotron2 loss"},
    )


@register_config("ofasys.criterion", "speech_pretrain_loss", SpeechPretrainCriterionConfig)
class SpeechPretrainCriterion(BaseCriterion):
    """This criterion will compute masked audio models (MAM) loss and tacotron2 loss,
    and return a weighted sum of these two.
    """

    def __init__(self, task, cfg: SpeechPretrainCriterionConfig):
        super().__init__(task, cfg)
        self.pred_masked_weight = cfg.pred_masked_weight
        self.pred_nomask_weight = cfg.pred_nomask_weight
        self.loss_weights = cfg.loss_weights
        self.log_keys = [] if cfg.log_keys is None else cfg.log_keys
        self.mam_weight = cfg.mam_weight
        self.dec_weight = cfg.dec_weight
        self.ctc_weight = cfg.ctc_weight
        self.sentence_avg = cfg.sentence_avg
        self.blank_idx = 0
        self.dict_start = task.target_dictionary.index("<phone>_dict_begin")
        self.dict_end = task.target_dictionary.index("<phone>_dict_end")

        self.speech_criterion = OFATacotron2Criterion(task, cfg)

    def forward(self, model, sample, reduction="sum", update_num=None, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.mam_weight is not None and self.mam_weight > 0.0:
            raw_sample = copy.deepcopy(sample)
            sample = build_mask_sample(sample, model)
        feat_out, extra, encoder_out = model(**sample["net_input"], return_encoder_out=True)
        # mam loss
        # mask_audio_prediction mode
        mam_loss = 0.0
        if self.mam_weight is not None and self.mam_weight > 0.0:
            with torch.no_grad():
                encoder_input = list(filter(lambda slot: slot.is_src, raw_sample["net_input"]["slots"]))
                teacher_encoder_out = model.encoder(encoder_input)
                teacher_enc = teacher_encoder_out["encoder_out"][0].transpose(0, 1)
                emb_weight = model.decoder.adaptor.embed_tokens.weight[self.dict_start : self.dict_end, :]
                emb_weight = emb_weight.detach()
                mam_teacher = F.linear(teacher_enc, emb_weight, None)

            student_enc = encoder_out["encoder_out"][0].transpose(0, 1)
            emb_weight = emb_weight.detach()
            mam_student = F.linear(student_enc, emb_weight, None)

            mask_indices = sample["mask_indices"]
            mam_student = mam_student[mask_indices]
            mam_teacher = mam_teacher[mask_indices]
            targ_logits = self.get_logits_for_ctc(mam_teacher)  # , dict_start, dict_end, blank_id
            pred_logits = self.get_logits_for_ctc(mam_student)  # , dict_start, dict_end, blank_id
            mam_loss = F.kl_div(
                utils.log_softmax(pred_logits.float(), dim=-1),
                utils.softmax(targ_logits.float(), dim=-1),
                reduction=reduction,
            )

        sample_size = sample["target"].size(0) if self.sentence_avg else targ_logits.shape[0]

        logging_output = {
            "mam_loss": mam_loss,
            "ntokens": targ_logits.shape[0],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.dec_weight == 0.0:
            loss = mam_loss
            logging_output["loss"] = loss.item()
            return loss, sample_size, logging_output

        ## dec loss
        bsz, max_len, _ = sample["target"].size()
        feat_tgt = sample["target"]
        feat_len = sample["target_lengths"].view(bsz, 1).expand(-1, max_len)
        eos_tgt = torch.arange(max_len).to(sample["target"].device)
        eos_tgt = eos_tgt.view(1, max_len).expand(bsz, -1)
        eos_tgt = (eos_tgt == (feat_len - 1)).float()
        tgt_lens = sample["target_lengths"]
        eos_out = extra["eos_out"]
        net_input = sample["net_input"]
        for slot in net_input['slots']:
            if slot.modality == ModalityType.AUDIO and slot.is_src:
                src_tokens = slot.value["fbank"]
                src_lens = slot.value["fbank_lengths"]

        l1_loss, mse_loss, eos_loss = self.speech_criterion.compute_loss(
            extra["feature_out"],
            feat_out,
            eos_out,
            feat_tgt,
            eos_tgt,
            tgt_lens,
            reduction="mean",
        )

        attn_loss = torch.tensor(0.0).type_as(l1_loss)
        if self.speech_criterion.guided_attn is not None:
            attn_loss = self.speech_criterion.guided_attn(extra["attn"], src_lens, tgt_lens, reduction="mean")

        dec_loss = l1_loss + mse_loss + eos_loss + attn_loss

        # Log tts loss
        logging_output['mam_loss'] = mam_loss.item()
        logging_output['dec_loss'] = dec_loss.item() * sample_size
        logging_output['l2_loss'] = mse_loss.item() * sample_size
        logging_output['l1_loss'] = l1_loss.item() * sample_size
        logging_output['bce_loss'] = eos_loss.item() * sample_size

        loss = self.mam_weight * mam_loss + self.dec_weight * dec_loss * sample_size
        logging_output["loss"] = loss.item()

        return loss * self.weight, sample_size, logging_output

    def get_logits_for_ctc(self, value, dict_start=None, dict_end=None, blank_id=1):
        logits = value
        # if dict_start is not None and dict_end is not None:
        #     phone_logits = logits[:, dict_start:dict_end]
        #     blank_logits = logits[:, blank_id:blank_id + 1]
        #     logits = torch.cat([blank_logits, phone_logits], dim=-1)
        return logits

    @classmethod
    def reduce_metrics(cls, logging_outputs, prefix_name=None) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        task_name = prefix_name + '/' if prefix_name else ''

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mam_loss_sum = sum(log.get("mam_loss", 0) for log in logging_outputs)
        dec_loss_sum = sum(log.get("dec_loss", 0) for log in logging_outputs)
        l1_loss_sum = sum(log.get("l1_loss", 0) for log in logging_outputs)
        l2_loss_sum = sum(log.get("l2_loss", 0) for log in logging_outputs)
        bce_loss_sum = sum(log.get("bce_loss", 0) for log in logging_outputs)

        # TODO: check how to compute numel of multi-loss
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, priority=0, round=3)
        metrics.log_scalar(f"{task_name}loss", loss_sum / sample_size, sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar(f"{task_name}nll_loss", loss_sum / ntokens, ntokens, round=3)
            metrics.log_derived(
                f"{task_name}ppl", lambda meters: utils.get_perplexity(meters[f"{task_name}nll_loss"].avg)
            )
        else:
            metrics.log_derived(f"{task_name}ppl", lambda meters: utils.get_perplexity(meters[f"{task_name}loss"].avg))

        metrics.log_scalar(f"{task_name}mam_loss", mam_loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar(f"{task_name}dec_loss", dec_loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar(f"{task_name}l1_loss", l1_loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar(f"{task_name}l2_loss", l2_loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar(f"{task_name}bce_loss", bce_loss_sum / sample_size, sample_size, round=5)
        if "enc_dec_attn_loss" in logging_outputs[0]:
            enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                f"{task_name}enc_dec_attn_loss", enc_dec_attn_loss_sum / sample_size, sample_size, round=8
            )

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return False
