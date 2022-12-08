# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ofasys.configure import register_config
from ofasys.utils.file_utils import cached_path

from .base import BaseMetric, MetricConfig
from .pyciderevalcap.ciderD.ciderD import CiderD


@dataclass
class CiderConfig(MetricConfig):
    cached_tokens: Optional[str] = field(
        default=None, metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"}
    )


@register_config("ofasys.metric", "cider", CiderConfig)
class Cider(BaseMetric):
    def __init__(self, cfg: CiderConfig):
        super().__init__(cfg)
        if cfg.cached_tokens is None or not cfg.cached_tokens or cfg.cached_tokens == 'corpus':
            cfg.cached_tokens = 'corpus'
            local_path = 'corpus'
        else:
            local_path = cached_path(cfg.cached_tokens)
        self.CiderD_scorer = CiderD(df=local_path)

    def compute(self, hyps: List[str], refs: List[List[str]]) -> Dict:
        assert isinstance(hyps, list) and isinstance(hyps[0], str)
        assert isinstance(refs, list) and isinstance(refs[0], list)
        assert isinstance(refs[0][0], str)
        assert len(hyps) == len(refs), "hyps({}) vs refs({})".format(len(hyps), len(refs))

        logging_output = {}
        gen_res_size = len(hyps)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [hyps[i].strip()]

        gts = OrderedDict()
        gt_res_ = [[refs[i][j].strip() for j in range(len(refs[i]))] for i in range(len(refs))]
        for i in range(gen_res_size):
            gts[i] = gt_res_[i]

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        _, scores = self.CiderD_scorer.compute_score(gts, res_)

        logging_output["_cider_score_sum"] = scores.sum()
        logging_output["_cider_cnt"] = scores.size
        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        def compute_cider(meters):
            cider = meters["_cider_score_sum"].sum / meters["_cider_cnt"].sum
            cider = cider if isinstance(cider, float) else cider.item()
            return round(cider, 3)

        if self.sum_logs(logging_outputs, "_cider_cnt") > 0:
            self.metrics.log_scalar("_cider_score_sum", self.sum_logs(logging_outputs, "_cider_score_sum"))
            self.metrics.log_scalar("_cider_cnt", self.sum_logs(logging_outputs, "_cider_cnt"))
            self.metrics.log_derived("cider", compute_cider)
