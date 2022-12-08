# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ofasys.configure import register_config
from ofasys.utils.file_utils import cached_path

from .base import BaseMetric, MetricConfig


@dataclass
class CLIPConfig(MetricConfig):
    clip_model: Optional[str] = field(default=None, metadata={"help": "name or path of clip model."})


@register_config("ofasys.metric", "clip_ti", CLIPConfig)
class CLIPTISim(BaseMetric):
    def __init__(self, cfg: CLIPConfig):
        super().__init__(cfg)
        # Import inside the metric for reducing the import time
        # Remove this if there is other place using the `clip` module
        import clip

        self.clip = clip
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        local_path = cached_path(cfg.clip_model)
        clip_model, clip_preprocess = clip.load(local_path)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_model.to(self.device)
        self.clip_model.eval()

    def compute(self, hyps, refs) -> Dict:
        clip_images_input = torch.stack([self.clip_preprocess(hyp_image) for hyp_image in hyps], dim=0).to(self.device)
        clip_text_input = self.clip.tokenize(refs).to(self.device)
        with torch.no_grad():
            hyp_image_features = self.clip_model.encode_image(clip_images_input)
            hyp_image_features /= hyp_image_features.norm(dim=-1, keepdim=True)
            text_features = self.clip_model.encode_text(clip_text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        ti_similarity = hyp_image_features @ text_features.T
        scores, indices = ti_similarity.max(dim=0)
        logging_output = {}
        logging_output["_clip_ti_score_sum"] = sum(scores).item()
        logging_output["_clip_ti_score_cnt"] = len(scores)
        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        def sum_logs(key):
            import torch

            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_clip_ti_score_sum"].sum / meters["_clip_ti_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_clip_ti_score_cnt") > 0:
            # log counts as numpy arrays -- log_scalar will sum them correctly
            self.metrics.log_scalar("_clip_ti_score_sum", sum_logs("_clip_ti_score_sum"))
            self.metrics.log_scalar("_clip_ti_score_cnt", sum_logs("_clip_ti_score_cnt"))
            self.metrics.log_derived("clip_ti", compute_score)
