# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any, Dict, List, Union

from ofasys.configure import register_config
from ofasys.generator import MultiGeneratorOutput, SequenceGeneratorOutput
from ofasys.preprocessor.default.text import remove_punctuation
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "caption", dataclass=TaskConfig)
class CaptionTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        if data.get('cap', None) is not None:
            caption = data['cap'].lower()
            if split == 'train' and not self.cfg.scst:
                caption = remove_punctuation(caption).strip()
                caption_token_list = caption.strip().split()
                caption = ' '.join(caption_token_list[: self.cfg.max_tgt_length])
            else:
                caption = ' '.join(caption.strip().split())
                caption_list = [remove_punctuation(cap).strip() for cap in caption.strip().split('&&')]
                caption = '&&'.join(caption_list)
                data['cap_list'] = caption_list
            data['cap'] = caption

        return data

    def inference(self, model, samples, **kwargs):
        hyps = super().inference(model, samples, **kwargs)

        multi_hyps: Union[SequenceGeneratorOutput, MultiGeneratorOutput]
        for multi_hyps in hyps:
            if isinstance(multi_hyps, List):
                for hyp in multi_hyps:
                    hyp.text = remove_punctuation(hyp.text).strip()
            else:
                multi_hyps.text = remove_punctuation(multi_hyps.text).strip()
        return hyps


@register_config("ofasys.task", "pretrain_caption", dataclass=TaskConfig)
class PretrainCaptionTask(OFATask):
    def __init__(self, cfg: TaskConfig):
        super().__init__(cfg)

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        if split == 'test':
            data['cap'] = 'dummy'
            return data
        caption = data['cap'].lower()

        caption = caption.strip()
        caption_token_list = caption.strip().split()
        caption = ' '.join(caption_token_list[:self.cfg.max_tgt_length])

        if len(caption) == 0:
            return None

        data['cap'] = caption
        return data

    def inference(self, model, samples):
        hyps = super().inference(model, samples)
        hyps = [remove_punctuation(hyp).strip() for hyp in hyps]
        return hyps
