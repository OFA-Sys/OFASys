# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import Dict

import torch

from ofasys.configure import register_config

from .base import BaseMetric, MetricConfig


@dataclass
class IouAccuracyConfig(MetricConfig):
    threshold: float = field(default=0.5, metadata={"help": "iou threshold"})


@register_config("ofasys.metric", "iou_acc", IouAccuracyConfig)
class IouAccuracy(BaseMetric):
    def __init__(self, cfg: IouAccuracyConfig):
        super().__init__(cfg)
        self.threshold = cfg.threshold

    def compute(self, hyps, refs) -> Dict:
        if not isinstance(refs, torch.Tensor):
            refs = torch.tensor(
                [list(map(float, ref.split(','))) for ref in refs],
                dtype=torch.float32,
            )
        hyps = torch.stack(hyps, dim=0).type_as(refs)
        hyps = hyps.float()
        refs = refs.float()

        interacts = torch.cat(
            [
                torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
                torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:]),
            ],
            dim=1,
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        scores = ((ious >= self.threshold) & (interacts_w > 0) & (interacts_h > 0)).float()

        logging_output = {}
        logging_output["_score_sum"] = scores.sum().item()
        logging_output["_score_cnt"] = scores.size(0)
        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        def sum_logs(key):
            import torch

            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if sum_logs("_score_cnt") > 0:
            # log counts as numpy arrays -- log_scalar will sum them correctly
            self.metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            self.metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))

            def compute_acc(meters):
                score = meters["_score_sum"].sum / meters["_score_cnt"].sum
                score = score if isinstance(score, float) else score.item()
                return round(score, 4)

            self.metrics.log_derived("iou_acc", compute_acc)
