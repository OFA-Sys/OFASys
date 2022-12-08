# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Dict

import torch

from ofasys.configure import register_config

from .base import BaseMetric, MetricConfig


@register_config("ofasys.metric", "accuracy", MetricConfig)
class Accuracy(BaseMetric):
    def __init__(self, cfg: MetricConfig):
        super().__init__(cfg)

    def compute(self, hyps, refs) -> Dict:
        logging_output = {}
        scores = [1.0 if str(ref) == str(hyp) else 0.0 for ref, hyp in zip(refs, hyps)]
        logging_output["_score_sum"] = sum(scores)
        logging_output["_score_cnt"] = len(scores)
        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        def sum_logs(key):
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

            self.metrics.log_derived("acc", compute_acc)
