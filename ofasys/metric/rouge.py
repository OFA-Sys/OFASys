# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
from typing import Dict

from ofasys.configure import register_config

from .base import BaseMetric, MetricConfig


@register_config("ofasys.metric", "rouge", MetricConfig)
class Rouge(BaseMetric):
    def __init__(self, cfg: MetricConfig):
        super().__init__(cfg)
        from datasets import load_metric

        self.rouge_metric = load_metric(os.path.join(os.path.dirname(__file__), '../../ofasys/utils/rouge.py'))

    def compute(self, hyps, refs) -> Dict:
        result = self.rouge_metric.compute(predictions=hyps, references=refs, use_agregator=False, use_stemmer=True)
        result_f1 = {key: sum([item.fmeasure for item in value]) * 100 for key, value in result.items()}

        logging_output = {}
        logging_output['_rouge1_f1_sum'] = result_f1['rouge1']
        logging_output['_rouge2_f1_sum'] = result_f1['rouge2']
        logging_output['_rougeL_f1_sum'] = result_f1['rougeL']
        logging_output['_rouge_cnt'] = len(hyps)
        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        def sum_logs(key):
            import torch

            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if sum_logs("_rouge_cnt") > 0:
            self.metrics.log_scalar("_rouge1_f1_sum", sum_logs("_rouge1_f1_sum"))
            self.metrics.log_scalar("_rouge2_f1_sum", sum_logs("_rouge2_f1_sum"))
            self.metrics.log_scalar("_rougeL_f1_sum", sum_logs("_rougeL_f1_sum"))
            self.metrics.log_scalar("_rouge_cnt", sum_logs("_rouge_cnt"))
            self.metrics.log_derived("rouge1_f1", lambda x: x["_rouge1_f1_sum"].sum / x["_rouge_cnt"].sum)
            self.metrics.log_derived("rouge2_f1", lambda x: x["_rouge2_f1_sum"].sum / x["_rouge_cnt"].sum)
            self.metrics.log_derived("rougeL_f1", lambda x: x["_rougeL_f1_sum"].sum / x["_rouge_cnt"].sum)
