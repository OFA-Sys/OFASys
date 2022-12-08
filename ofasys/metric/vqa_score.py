# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Dict

import torch

from ofasys.configure import register_config

from .base import BaseMetric, MetricConfig


@register_config("ofasys.metric", "vqa_score", MetricConfig)
class VqaScore(BaseMetric):
    def __init__(self, cfg: MetricConfig):
        super().__init__(cfg)

    def compute(self, hyps, refs) -> Dict:
        logging_output = {}
        scores = [ref.get(hyp, 0) for ref, hyp in zip(refs, hyps)]
        logging_output["_vqa_score_sum"] = sum(scores)
        logging_output["_vqa_cnt"] = len(scores)
        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        def sum_logs(key):
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_vqa_score_sum"].sum / meters["_vqa_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_vqa_cnt") > 0:
            self.metrics.log_scalar("_vqa_score_sum", sum_logs("_vqa_score_sum"))
            self.metrics.log_scalar("_vqa_cnt", sum_logs("_vqa_cnt"))
            self.metrics.log_derived("vqa_score", compute_score)
