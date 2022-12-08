# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from ofasys.configure import BaseDataclass


@dataclass
class CriterionConfig(BaseDataclass):
    is_active: bool = field(default=False, metadata={"help": "is active for config_store"})
    weight: float = field(default=1.0, metadata={"help": "the weight of loss item"})


class BaseCriterion(torch.nn.Module):
    """
    Criterion module is responsible for calling the model and calculating the loss.
    """

    def __init__(self, task, cfg: CriterionConfig = None):
        """
        Args:
            task: the task corresponding to the criterion.
            cfg (CriterionConfig): the config to the criterion.
        """
        super().__init__()
        self.cfg = cfg
        self.weight = cfg.weight
        self.task = task
        if hasattr(task, "target_dictionary"):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100

    def forward(self, model, sample, update_num=0, reduce=True):
        """
        Calling the model, calculating the loss and prepare `logging_output` for metrics.

        Args:
            model: the model for criterion.
            sample (Dict[str, Any]): the batched samples for calculating loss.
            update_num (Int): the number of current update steps, default: 0.
            reduce (Bool): if ``true``, it will return the sum of losses.
                Otherwise, it will return the loss item for each sample. default: ``true``.
        """
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]], prefix_name=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return False
