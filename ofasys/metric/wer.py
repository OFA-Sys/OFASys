# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ofasys.configure import register_config
from ofasys.preprocessor.tokenizer.evaluate_tokenizer import EvaluationTokenizer
from ofasys.utils.file_utils import cached_path

from .base import BaseMetric, MetricConfig


@dataclass
class WerConfig(MetricConfig):
    wer_tokenizer: EvaluationTokenizer.ALL_TOKENIZER_TYPES = field(
        default="none", metadata={"help": "sacreBLEU tokenizer to use for evaluation"}
    )
    wer_remove_punct: bool = field(default=False, metadata={"help": "remove punctuation"})
    wer_char_level: bool = field(default=False, metadata={"help": "evaluate at character level"})
    wer_lowercase: bool = field(default=False, metadata={"help": "lowercasing"})
    ref_chinese_split: bool = field(default=False, metadata={"help": "chinese"})
    pred_chinese_split: bool = field(default=False, metadata={"help": "chinese"})


@register_config("ofasys.metric", "wer", WerConfig)
class Wer(BaseMetric):
    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")
        self.ed = ed
        self.ref_chinese_split = cfg.ref_chinese_split
        self.pred_chinese_split = cfg.pred_chinese_split
        self.tokenizer = EvaluationTokenizer(
            tokenizer_type=self.cfg.wer_tokenizer,
            lowercase=self.cfg.wer_lowercase,
            punctuation_removal=self.cfg.wer_remove_punct,
            character_tokenization=self.cfg.wer_char_level,
        )

    def compute(self, hyps: List[str], refs: List[List[str]]) -> Dict:
        assert isinstance(hyps, list) and isinstance(hyps[0], str)
        assert isinstance(refs, list) and isinstance(refs[0], str)
        assert len(hyps) == len(refs), "hyps({}) vs refs({})".format(len(hyps), len(refs))

        logging_output = {}
        gen_res_size = len(hyps)
        distance = 0
        ref_length = 0

        for i in range(gen_res_size):
            ref = refs[i]
            pred = hyps[i]
            ref_items = self.tokenizer.tokenize(ref).split()
            if self.ref_chinese_split:
                ref_items = [x for x in " ".join(ref_items)]
            pred_items = self.tokenizer.tokenize(pred).split()
            if self.pred_chinese_split:
                pred_items = [x for x in " ".join(pred_items)]
            distance += self.ed.eval(ref_items, pred_items)
            ref_length += len(ref_items)

        logging_output["_wer_score_sum"] = distance
        logging_output["_wer_cnt"] = ref_length

        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        def sum_logs(key):
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_wer(meters):
            wer = meters["_wer_score_sum"].sum / meters["_wer_cnt"].sum
            wer = wer if isinstance(wer, float) else wer.item()
            return round(wer, 3)

        if self.sum_logs(logging_outputs, "_wer_cnt") > 0:
            self.metrics.log_scalar("_wer_score_sum", sum_logs("_wer_score_sum"))
            self.metrics.log_scalar("_wer_cnt", sum_logs("_wer_cnt"))
            self.metrics.log_derived("wer", compute_wer)
