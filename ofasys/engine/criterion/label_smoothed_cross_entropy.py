# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from ofasys.configure import register_config
from ofasys.logging import metrics
from ofasys.module import utils

from .base import BaseCriterion, CriterionConfig


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(CriterionConfig):
    """
    Args:
        label_smoothing (Float): epsilon for label smoothing. Default: 0.0.
        report_accuracy (Bool): whether to report accuracy metrics. Default: ``false``.
        ignore_prefix_size (Int): ignore first N tokens. Default: 0.
        sentence_avg (Bool): if ``true``, the gradient will be normalized by the number of sentences.
        drop_worst_ratio (Float): when ``update_num > drop_worst_after``, the ``drop_worst_ratio * 100%`` worst sample will be discarded.
        drop_worst_after (Int): steps for discarding bad samples.
        constraint_range (Optional[str]): only `[constraint_start, constraint_end)` range in the vocabulary is involved in loss calculation.

    """

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
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
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


@register_config("ofasys.criterion", "label_smoothed_cross_entropy", LabelSmoothedCrossEntropyCriterionConfig)
class LabelSmoothedCrossEntropyCriterion(BaseCriterion):
    """This criterion will compute label-smoothed cross entropy loss and return it."""

    def __init__(self, task, cfg: CriterionConfig):
        super().__init__(task, cfg)
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

    def forward(self, model, sample, update_num=0, reduce=True):
        """
        Compute the loss for the given sample.

        Args:
            model: the model for criterion.
            sample (Dict[str, Any]): the batched samples for calculating loss.
            update_num (Int): the number of current update steps, default: 0.
            reduce (Bool): if ``true``, it will return the sum of losses.
                Otherwise, it will return the loss item for each sample. default: ``true``.

        Returns:
            loss: the calculated loss
            sample_size: this will be used as the denominator for the gradient
            logging_output: logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss * self.weight, sample_size, logging_output

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

    def get_lprobs_and_target(self, model, net_output, sample):
        constraint_masks = self.get_constraint_masks(net_output, sample)
        if constraint_masks is not None:
            net_output = (net_output[0].masked_fill(~constraint_masks, -math.inf),) + net_output[1:]

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size :, :].contiguous()

        if constraint_masks is not None:
            constraint_masks = constraint_masks.reshape(-1, constraint_masks.size(-1))
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
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, priority=0, round=3)
        metrics.log_scalar(f"{task_name}loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar(f"{task_name}nll_loss", nll_loss_sum / sample_size, ntokens, round=3)
        metrics.log_derived(f"{task_name}ppl", lambda meters: utils.get_perplexity(meters[f"{task_name}nll_loss"].avg))

        metrics.log_scalar(f"{task_name}ntokens", ntokens, 1, round=3)
        metrics.log_scalar(f"{task_name}bsz", nsentences, 1, round=3)
        metrics.log_scalar(f"{task_name}sample_size", sample_size, 1, round=3)

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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
