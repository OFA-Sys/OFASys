# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

import torch.nn.functional as F

from ofasys.configure import register_config
from ofasys.logging import metrics
from ofasys.module import utils

from .base import BaseCriterion, CriterionConfig


@dataclass
class CrossEntropyCriterionConfig(CriterionConfig):
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
    )


def nll_loss(lprobs, target, ignore_index=None, reduce=True):
    """Like torch.nn.functional.nll_loss but works for large inputs."""
    if lprobs.numel() < 2e9:
        return F.nll_loss(lprobs, target, ignore_index=ignore_index, reduction="sum" if reduce else "none")
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
    return nll_loss


@register_config("ofasys.criterion", "cross_entropy", CrossEntropyCriterionConfig)
class CrossEntropyCriterion(BaseCriterion):
    def __init__(self, task, cfg: CriterionConfig):
        super().__init__(task, cfg)
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, update_num=0, reduce=True):
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = nll_loss(lprobs, target, ignore_index=self.padding_idx, reduce=reduce)
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs, prefix_name=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        task_name = prefix_name + '/' if prefix_name else ''
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, priority=0, round=3)
        metrics.log_scalar(f"{task_name}loss", loss_sum / sample_size, sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar(f"{task_name}nll_loss", loss_sum / ntokens, ntokens, round=3)
            metrics.log_derived(
                f"{task_name}ppl", lambda meters: utils.get_perplexity(meters[f"{task_name}nll_loss"].avg)
            )
        else:
            metrics.log_derived(f"{task_name}ppl", lambda meters: utils.get_perplexity(meters[f"{task_name}loss"].avg))
        metrics.log_scalar(f"{task_name}ntokens", ntokens, 1, round=3)
        metrics.log_scalar(f"{task_name}bsz", nsentences, 1, round=3)
        metrics.log_scalar(f"{task_name}sample_size", sample_size, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
