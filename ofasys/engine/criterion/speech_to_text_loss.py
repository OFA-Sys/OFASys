# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ofasys.configure import register_config
from ofasys.logging import metrics
from ofasys.logging.meters import safe_round
from ofasys.module import utils
from ofasys.preprocessor.data_utils import post_process

from .base import BaseCriterion, CriterionConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeechtoTextLossConfig(CriterionConfig):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
    )
    post_process: Optional[str] = field(
        default="sentencepiece",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See ofasys.preprocessor.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={"help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"},
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={"help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"},
    )

    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )

    ce_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for cross entropy"},
    )
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "loss weiehgt for ctc in ASR"},
    )
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    constraint_range: Optional[str] = field(default=None, metadata={"help": "constraint range"})


def label_smoothed_nll_loss(
    lprobs,
    target,
    epsilon,
    update_num,
    reduce=True,
    drop_worst_ratio=0.0,
    drop_worst_after=0,
    constraint_masks=None,
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if constraint_masks is not None:
        smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        loss, indices = torch.topk(loss, k=int(loss.shape[0] * (1 - drop_worst_ratio)), largest=False)
        nll_loss = nll_loss[indices]
        lprobs = lprobs[indices]

    ntokens = loss.numel()
    nll_loss = nll_loss.sum()
    loss = loss.sum()

    return loss, nll_loss, ntokens


@register_config("ofasys.criterion", "speech_to_text_loss", SpeechtoTextLossConfig)
class SpeechtoTextLoss(BaseCriterion):
    """Criterion for speech_to_text task.
    This criterion will compute label smoothed cross entropy loss and CTC loss,
    and return a weighted sum of these two.
    """

    def __init__(self, task, cfg: SpeechtoTextLossConfig):
        super().__init__(task, cfg)

        self.blank_idx = 0
        self.dict_start = task.target_dictionary.index("<phone>_dict_begin")
        self.dict_end = task.target_dictionary.index("<phone>_dict_end")

        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process
        self.ce_weight = cfg.ce_weight
        self.ctc_weight = cfg.ctc_weight

        ## for ce
        self.sentence_avg = cfg.sentence_avg
        self.eps = cfg.label_smoothing
        self.ignore_prefix_size = cfg.ignore_prefix_size
        self.report_accuracy = cfg.report_accuracy

        self.drop_worst_ratio = cfg.drop_worst_ratio
        self.drop_worst_after = cfg.drop_worst_after

        self.constraint_start = None
        self.constraint_end = None
        if cfg.constraint_range is not None:
            constraint_start, constraint_end = cfg.constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity

        if self.ce_weight > 0 and self.ctc_weight > 0:
            logger.info("Using cross entropy loss and CTC loss for ASR")
        elif self.ce_weight > 0:
            logger.info("Only using CE loss")
        elif self.ctc_weight > 0:
            logger.info("Only using CTC loss for ASR")
        else:
            logger.info("ERROR")

    def forward(self, model, sample, update_num=0, reduce=True):

        net_output = model(**sample["net_input"], return_encoder_out=True)

        if self.ce_weight > 0:
            loss_ce, nll_loss_ce, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)
        else:
            nll_loss_ce = None
        extra, encoder_out = net_output[1], net_output[2]
        x_for_ctc = None

        if self.ctc_weight is not None and self.ctc_weight > 0.0:
            emb_weight = model.decoder.adaptor.embed_tokens.weight[self.dict_start : self.dict_end, :]
            if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
                enc = encoder_out["encoder_out"][0]
                x_for_ctc = F.linear(enc, emb_weight, None)

        if x_for_ctc is not None:
            extra["encoder_out_for_ctc"] = [x_for_ctc]  # T x B x C
            extra["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]

        if self.ctc_weight > 0:
            loss_ctc, lprobs, input_lengths = self.compute_loss_ctc(model, extra, sample)

        if self.ce_weight > 0 and self.ctc_weight > 0:
            loss = self.ce_weight * loss_ce + self.ctc_weight * loss_ctc
        elif self.ce_weight > 0:
            loss = loss_ce
        elif self.ctc_weight > 0:
            loss = loss_ctc
        else:
            logger.info("ERROR: must ce_weight > 0 or ctc_weight > 0")

        ntokens = sample["ntokens"] if "ntokens" in sample else sample["target_lengths"].sum().item()

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        logging_output = {
            "loss": loss.item(),
            "ce_loss": loss_ce.item() if self.ce_weight > 0 else 0,
            "ctc_loss": loss_ctc.item() if self.ctc_weight > 0 else 0,
            "nll_loss": nll_loss_ce.item() if nll_loss_ce is not None else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.ce_weight > 0 and self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.item())
            logging_output["total"] = utils.item(total.data)

        # compute Word Error Rate (WER)
        if self.ctc_weight > 0 and self.report_accuracy and not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["encoder_target"] if "encoder_target" in sample else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (t != self.task.target_dictionary.eos())
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx]
                    pred_units_arr = pred_units_arr + self.dict_start - 1
                    pred_units_arr = pred_units_arr.tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss * self.weight, sample_size, logging_output

    def get_normalized_probs_for_ctc(self, net_output, log_probs, dict_start=None, dict_end=None, blank_id=1):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out_for_ctc"][0]  # T x B x C
        # if dict_start is not None and dict_end is not None:
        #     phone_logits = logits[:, :, dict_start:dict_end]
        #     blank_logits = logits[:, :, blank_id:blank_id+1]
        #     logits = torch.cat([blank_logits, phone_logits], dim=-1)
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def compute_loss_ctc(self, model, net_output, sample):
        # self.dict_start = sample["dict_start"]
        # self.dict_end = sample["dict_end"]
        # blank_id = sample["blank_id"]
        lprobs = self.get_normalized_probs_for_ctc(
            net_output, log_probs=True  # , dict_start=self.dict_start, dict_end=self.dict_end, blank_id=blank_id
        ).contiguous()  # (T, B, C) from the encoder

        if net_output["encoder_padding_mask"] is not None:
            non_padding_mask = ~net_output["encoder_padding_mask"][0]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full((lprobs.size(1),), lprobs.size(0), dtype=torch.long)

        pad_mask = (sample["encoder_target"] != self.padding_idx) & (sample["encoder_target"] != self.eos_idx)
        # targets = sample["encoder_target"] - self.dict_start + 1
        targets = sample["encoder_target"] - self.dict_start
        targets_flat = targets.masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
            ##processing
            target_lengths = target_lengths - 1
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss_ctc = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        assert (targets_flat >= 0).all() and (targets_flat <= (self.dict_end - self.dict_start)).all(), print(
            targets_flat, 0, self.dict_end - self.dict_start, loss_ctc
        )

        return loss_ctc, lprobs, input_lengths

    def get_constraint_masks(self, net_output, sample):
        if self.constraint_start is not None and self.constraint_end is not None:
            constraint_masks = torch.ones(net_output[0].shape, dtype=torch.bool, device=net_output[0].device)
            constraint_masks[..., 4 : self.constraint_start] = 0
            constraint_masks[..., self.constraint_end :] = 0
            if sample.get("constraint_masks", None) is not None:
                constraint_masks = torch.logical_and(sample["constraint_masks"], constraint_masks)
            return constraint_masks
        else:
            return sample.get("constraint_masks", None)

    ## for ce
    def get_lprobs_and_target(self, model, net_output, sample):
        constraint_masks = self.get_constraint_masks(net_output, sample)
        net_output = (net_output[0].masked_fill(~constraint_masks, -math.inf),) + net_output[1:]

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size :, :]
                constraint_masks = constraint_masks.contiguous()

        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), constraint_masks

    def compute_loss(self, model, net_output, sample, update_num, reduce=True):
        lprobs, target, constraint_masks = self.get_lprobs_and_target(model, net_output, sample)
        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != self.padding_idx]
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx]
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            constraint_masks=constraint_masks,
        )
        return loss, nll_loss, ntokens

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target, _ = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs, prefix_name=None) -> None:
        """Aggregate logging outputs from data parallel training."""

        task_name = prefix_name + '/' if prefix_name else ''
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, priority=0, round=3)
        metrics.log_scalar(f"{task_name}loss", loss_sum / sample_size, sample_size, round=3)

        metrics.log_scalar(f"{task_name}ctc_loss", ctc_loss_sum / sample_size, round=3)
        metrics.log_scalar(f"{task_name}ce_loss", ce_loss_sum / sample_size, round=3)
        metrics.log_scalar(f"{task_name}nll_loss", nll_loss_sum / sample_size, round=3)
        metrics.log_derived(f"{task_name}ppl", lambda meters: utils.get_perplexity(meters[f"{task_name}nll_loss"].avg))

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar(f"{task_name}total", total)
            n_correct = utils.item(sum(log.get("n_correct", 0) for log in logging_outputs))
            metrics.log_scalar(f"{task_name}n_correct", n_correct)
            metrics.log_derived(
                f"{task_name}accuracy",
                lambda meters: round(meters[f"{task_name}n_correct"].sum * 100.0 / meters[f"{task_name}total"].sum, 3)
                if meters[f"{task_name}total"].sum > 0
                else float("nan"),
            )

        metrics.log_scalar(f"{task_name}ntokens", ntokens)
        metrics.log_scalar(f"{task_name}nsentences", nsentences)

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar(f"{task_name}_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar(f"{task_name}_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar(f"{task_name}_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar(f"{task_name}_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar(f"{task_name}_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                f"{task_name}uer",
                lambda meters: safe_round(
                    meters[f"{task_name}_c_errors"].sum * 100.0 / meters[f"{task_name}_c_total"].sum, 3
                )
                if meters[f"{task_name}_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                f"{task_name}wer",
                lambda meters: safe_round(
                    meters[f"{task_name}_w_errors"].sum * 100.0 / meters[f"{task_name}_w_total"].sum, 3
                )
                if meters[f"{task_name}_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                f"{task_name}raw_wer",
                lambda meters: safe_round(
                    meters[f"{task_name}_wv_errors"].sum * 100.0 / meters[f"{task_name}_w_total"].sum, 3
                )
                if meters[f"{task_name}_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
